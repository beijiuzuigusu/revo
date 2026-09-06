"""Use the external visual force model to control the Revo2 index finger.

The only pressure feedback used by this module is the three-axis resultant
``sqrt(Fx**2 + Fy**2 + Fz**2)`` from
``force_indentify/checkpoints/array_skin.pth``.  Revo2's built-in tactile
sensor is deliberately outside this control path.

This is experimental software control, not a hardware emergency stop.  The
70/100 thresholds are uncalibrated model-output deltas, not N or Pa.  Real-hand
operation still requires a physical emergency stop, a clear workspace, and an
operator ready to remove power.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import math
import queue
import statistics
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol


DEFAULT_STOP_PRESSURE = 10.0
DEFAULT_RELIEF_PRESSURE = 20.0
DEFAULT_DOWN_SPEED = 100
DEFAULT_UP_SPEED = 50
DEFAULT_PORT_NAME = "COM11"
DEFAULT_EXPECTED_HAND = "right"
DEFAULT_CAMERA_INDEX = 1
DEFAULT_RESET_TIMEOUT_S = 5.0
DEFAULT_POLL_INTERVAL_S = 0.05
DEFAULT_MAX_SAMPLE_AGE_S = 1.0
BASELINE_VALID_SAMPLE_COUNT = 3
INDEX_RESET_POSITION = 0.0
FINGER_ID_TO_MOTOR_SLOT_OFFSET = 1
SDK_SPEED_MAX = 1000
REAL_MOTION_CONFIRMED = True

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
RESIZED_WIDTH = 640
RESIZED_HEIGHT = 360
ROI_COLUMN_START = 120
ROI_COLUMN_STOP = 480

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
FORCE_IDENTIFY_ROOT = _WORKSPACE_ROOT / "force_indentify"
ARRAY_SKIN_MODEL_PATH = FORCE_IDENTIFY_ROOT / "checkpoints" / "array_skin.pth"


class ControlError(RuntimeError):
    """A fail-closed control or integration error."""


class PressureSourceError(ControlError):
    """The pressure provider failed or did not produce a fresh sample."""


class PressureSampleTimeout(PressureSourceError):
    """The provider did not produce another sample before its deadline."""


class ControlState(str, Enum):
    RESETTING = "RESETTING"
    BASELINING = "BASELINING"
    OBSERVING = "OBSERVING"
    PRESSING = "PRESSING"
    HOLDING = "HOLDING"
    RELIEVING = "RELIEVING"


@dataclass(frozen=True)
class ControlConfig:
    stop_pressure: float = DEFAULT_STOP_PRESSURE
    relief_pressure: float = DEFAULT_RELIEF_PRESSURE
    down_speed: int = DEFAULT_DOWN_SPEED
    up_speed: int = DEFAULT_UP_SPEED
    reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S
    max_sample_age_s: float = DEFAULT_MAX_SAMPLE_AGE_S

    def validate(self) -> None:
        if not math.isfinite(self.stop_pressure) or self.stop_pressure < 0:
            raise ValueError("stop_pressure must be a non-negative finite value")
        if not math.isfinite(self.relief_pressure):
            raise ValueError("relief_pressure must be finite")
        if self.relief_pressure <= self.stop_pressure:
            raise ValueError("relief_pressure must be greater than stop_pressure")
        if not 1 <= self.down_speed <= SDK_SPEED_MAX:
            raise ValueError("down_speed must be in 1..1000")
        if not 1 <= self.up_speed <= SDK_SPEED_MAX:
            raise ValueError("up_speed must be in 1..1000")
        if not math.isfinite(self.reset_timeout_s) or self.reset_timeout_s <= 0:
            raise ValueError("reset_timeout_s must be greater than zero")
        if not math.isfinite(self.poll_interval_s) or self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be greater than zero")
        if not math.isfinite(self.max_sample_age_s) or self.max_sample_age_s <= 0:
            raise ValueError("max_sample_age_s must be greater than zero")


@dataclass(frozen=True)
class PressureSample:
    sequence: int
    timestamp: float
    fx: float
    fy: float
    fz: float

    @property
    def pressure_raw(self) -> float:
        return math.hypot(self.fx, self.fy, self.fz)


@dataclass(frozen=True)
class Telemetry:
    state: ControlState
    fx: float
    fy: float
    fz: float
    pressure_raw: float
    pressure_baseline: float | None
    pressure_control: float
    sample_timestamp: float
    sample_age_s: float
    position: float
    motor_state: str


class PressureProvider(Protocol):
    async def start(self) -> None:
        """Load/open resources and begin producing samples."""

    async def next_sample(self, timeout_s: float) -> PressureSample:
        """Return one newly produced sample, never the same frame twice."""

    async def close(self) -> None:
        """Stop production and release the camera/resources."""


TelemetryCallback = Callable[[Telemetry], None]
SleepFunction = Callable[[float], Awaitable[None]]
CleanupErrorCallback = Callable[[str], None]
ConnectHand = Callable[[], Awaitable[tuple[Any, int]]]
CloseHand = Callable[[Any], Awaitable[None]]


@dataclass(frozen=True)
class _WorkerFailure:
    error: BaseException


def _flatten_prediction(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        flattened: list[Any] = []
        for item in value:
            if isinstance(item, Sequence) and not isinstance(
                item, (str, bytes, bytearray)
            ):
                flattened.extend(_flatten_prediction(item))
            else:
                flattened.append(item)
        return flattened
    return [value]


def pressure_sample_from_prediction(
    prediction: Any,
    *,
    sequence: int,
    timestamp: float,
) -> PressureSample:
    """Validate a model result and establish the public pressure contract."""

    values = _flatten_prediction(prediction)
    if len(values) != 3:
        raise PressureSourceError(
            f"model output must contain exactly three values, got {len(values)}"
        )
    try:
        fx, fy, fz = (float(value) for value in values)
        sample_timestamp = float(timestamp)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PressureSourceError("model output cannot be converted to three numbers") from exc
    if not all(math.isfinite(value) for value in (fx, fy, fz)):
        raise PressureSourceError("model output contains NaN or Inf")
    if not math.isfinite(sample_timestamp):
        raise PressureSourceError("prediction timestamp is not finite")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise PressureSourceError("prediction sequence must be a non-negative integer")
    return PressureSample(sequence, sample_timestamp, fx, fy, fz)


def _load_cnn_module(models_root: Path) -> ModuleType:
    """Load only ``Convolutional.Cnn`` without importing unrelated models."""

    package_name = "_revo2_external_force_models"
    module_name = f"{package_name}.Convolutional"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    # A small synthetic package preserves Convolutional.py's relative import of
    # basic_module.py while avoiding models/__init__.py, which eagerly imports
    # every experimental architecture in force_indentify.
    package = sys.modules.get(package_name)
    if package is None:
        package = ModuleType(package_name)
        package.__path__ = [str(models_root)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package

    model_file = models_root / "Convolutional.py"
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise PressureSourceError(f"cannot load Cnn model module: {model_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


class CameraModelPressureProvider:
    """Latest-only camera/model provider with a non-blocking stale watchdog path."""

    def __init__(
        self,
        *,
        model_path: Path = ARRAY_SKIN_MODEL_PATH,
        camera_index: int = DEFAULT_CAMERA_INDEX,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.camera_index = camera_index
        self._monotonic = monotonic
        self._items: queue.Queue[PressureSample | _WorkerFailure] = queue.Queue(
            maxsize=1
        )
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: Any = None
        self._model: Any = None
        self._transform: Any = None
        self._device: Any = None
        self._cv2: Any = None
        self._torch: Any = None
        self._image_type: Any = None
        self._sequence = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            raise PressureSourceError("pressure provider has already been started")
        await asyncio.to_thread(self._open_and_start_worker)

    def _open_and_start_worker(self) -> None:
        if self.model_path != ARRAY_SKIN_MODEL_PATH.resolve():
            raise PressureSourceError(
                "production control only permits force_indentify/checkpoints/array_skin.pth"
            )
        if not self.model_path.is_file():
            raise PressureSourceError(f"model weights not found: {self.model_path}")

        try:
            import cv2
            import torch
            from PIL import Image
            from torchvision import transforms as T
        except Exception as exc:
            raise PressureSourceError(
                "camera/model dependencies are unavailable (cv2, torch, PIL, torchvision)"
            ) from exc

        models = _load_cnn_module(FORCE_IDENTIFY_ROOT / "models")
        model_type = getattr(models, "Cnn", None)
        if model_type is None:
            raise PressureSourceError("force_indentify.models does not expose Cnn")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model_type().to(device)
        try:
            state_dict = torch.load(
                str(self.model_path), map_location=device, weights_only=True
            )
        except TypeError:
            state_dict = torch.load(str(self.model_path), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        if not capture.isOpened():
            capture.release()
            raise PressureSourceError(
                f"cannot open camera {self.camera_index} with DirectShow"
            )

        self._cv2 = cv2
        self._torch = torch
        self._image_type = Image
        self._transform = T.Compose([T.ToTensor()])
        self._device = device
        self._model = model
        self._capture = capture
        self._started = True
        self._thread = threading.Thread(
            target=self._worker,
            name="revo2-external-pressure",
            daemon=True,
        )
        self._thread.start()

    def _worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    raise PressureSourceError("camera frame read failed")

                resized = self._cv2.resize(
                    frame, (RESIZED_WIDTH, RESIZED_HEIGHT)
                )
                roi = resized[:, ROI_COLUMN_START:ROI_COLUMN_STOP, :]
                if tuple(roi.shape[:2]) != (RESIZED_HEIGHT, RESIZED_HEIGHT):
                    raise PressureSourceError(
                        f"camera ROI has unexpected shape: {getattr(roi, 'shape', None)}"
                    )

                # Deliberately preserve OpenCV's BGR channel order.  This matches
                # the deployed force_indentify inference code and adds no
                # normalization beyond ToTensor().
                image = self._image_type.fromarray(roi)
                tensor = self._transform(image).unsqueeze(0).to(self._device)
                with self._torch.inference_mode():
                    prediction = self._model(tensor)
                sample = pressure_sample_from_prediction(
                    prediction,
                    sequence=self._sequence,
                    timestamp=self._monotonic(),
                )
                self._sequence += 1
                self._put_latest(sample)
        except BaseException as exc:
            if not self._stop_event.is_set():
                self._put_latest(_WorkerFailure(exc))

    def _put_latest(self, item: PressureSample | _WorkerFailure) -> None:
        while True:
            try:
                self._items.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._items.get_nowait()
                except queue.Empty:
                    pass

    async def next_sample(self, timeout_s: float) -> PressureSample:
        if not self._started:
            raise PressureSourceError("pressure provider is not started")
        if timeout_s < 0 or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be a non-negative finite value")
        try:
            if timeout_s == 0:
                item = self._items.get_nowait()
            else:
                item = await asyncio.to_thread(
                    self._items.get, True, timeout_s
                )
        except queue.Empty as exc:
            raise PressureSampleTimeout("no new prediction before deadline") from exc

        if isinstance(item, _WorkerFailure):
            raise PressureSourceError(f"pressure worker failed: {item.error}") from item.error
        return item

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        self._stop_event.set()
        capture = self._capture
        if capture is not None:
            capture.release()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._capture = None
        self._thread = None
        self._started = False


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _validate_sample(sample: Any) -> PressureSample:
    if not isinstance(sample, PressureSample):
        raise PressureSourceError(
            f"provider returned an invalid sample object: {type(sample).__name__}"
        )
    return pressure_sample_from_prediction(
        [sample.fx, sample.fy, sample.fz],
        sequence=sample.sequence,
        timestamp=sample.timestamp,
    )


async def _next_fresh_sample(
    provider: PressureProvider,
    config: ControlConfig,
    *,
    previous: PressureSample | None,
    monotonic: Callable[[], float],
) -> tuple[PressureSample, float]:
    now = monotonic()
    if previous is None:
        remaining = config.max_sample_age_s
    else:
        previous_age = now - previous.timestamp
        if previous_age < 0:
            raise PressureSourceError("prediction timestamp is ahead of monotonic time")
        if previous_age > config.max_sample_age_s:
            raise PressureSampleTimeout(
                f"latest prediction is stale ({previous_age:.3f}s)"
            )
        remaining = max(0.0, config.max_sample_age_s - previous_age)

    try:
        if remaining == 0:
            sample = await provider.next_sample(0)
        else:
            sample = await asyncio.wait_for(
                provider.next_sample(remaining), timeout=remaining
            )
    except (asyncio.TimeoutError, TimeoutError, PressureSampleTimeout) as exc:
        age = (
            monotonic() - previous.timestamp
            if previous is not None
            else config.max_sample_age_s
        )
        raise PressureSampleTimeout(
            f"no new valid prediction within {config.max_sample_age_s:.3f}s "
            f"(latest age {age:.3f}s)"
        ) from exc
    except PressureSourceError:
        raise
    except Exception as exc:
        raise PressureSourceError(f"pressure provider failed: {exc}") from exc

    sample = _validate_sample(sample)
    if previous is not None and sample.sequence <= previous.sequence:
        raise PressureSourceError(
            f"prediction was reused or out of order: {sample.sequence} <= {previous.sequence}"
        )
    age = monotonic() - sample.timestamp
    if age < 0:
        raise PressureSourceError("prediction timestamp is ahead of monotonic time")
    if age > config.max_sample_age_s:
        raise PressureSampleTimeout(f"prediction is stale ({age:.3f}s)")
    return sample, age


async def _set_index_speed(client: Any, slave_id: int, sdk: Any, speed: int) -> None:
    await client.set_finger_speed(slave_id, sdk.FingerId.Index, speed)


async def _read_index_motor(
    client: Any,
    slave_id: int,
    sdk: Any,
) -> tuple[float, str]:
    motor = await client.get_motor_status(slave_id)
    index = int(sdk.FingerId.Index) - FINGER_ID_TO_MOTOR_SLOT_OFFSET
    positions = getattr(motor, "positions", ())
    states = getattr(motor, "states", ())
    if index < 0 or index >= len(positions) or index >= len(states):
        raise ControlError(
            "motor status does not contain index-finger data: "
            f"finger_id={int(sdk.FingerId.Index)}, "
            f"positions={len(positions)}, states={len(states)}"
        )
    try:
        position = float(positions[index])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlError("index position cannot be converted to a number") from exc
    motor_state = str(states[index])
    if not math.isfinite(position):
        raise ControlError(f"index position is not finite: {position!r}")
    if "stall" in motor_state.casefold():
        raise ControlError(f"index motor reports a stall: {motor_state}")
    return position, motor_state


def _report(callback: TelemetryCallback | None, telemetry: Telemetry) -> None:
    if callback is not None:
        callback(telemetry)


def _make_telemetry(
    state: ControlState,
    sample: PressureSample,
    sample_age_s: float,
    position: float,
    motor_state: str,
    *,
    baseline: float | None,
    pressure_control: float = math.nan,
) -> Telemetry:
    return Telemetry(
        state=state,
        fx=sample.fx,
        fy=sample.fy,
        fz=sample.fz,
        pressure_raw=sample.pressure_raw,
        pressure_baseline=baseline,
        pressure_control=pressure_control,
        sample_timestamp=sample.timestamp,
        sample_age_s=sample_age_s,
        position=position,
        motor_state=motor_state,
    )


async def _reset_index_to_zero(
    client: Any,
    slave_id: int,
    sdk: Any,
    provider: PressureProvider,
    config: ControlConfig,
    previous_sample: PressureSample,
    *,
    on_telemetry: TelemetryCallback | None,
    sleep: SleepFunction,
    monotonic: Callable[[], float],
) -> PressureSample:
    await client.set_finger_position_with_speed(
        slave_id,
        sdk.FingerId.Index,
        int(INDEX_RESET_POSITION),
        config.up_speed,
    )
    deadline = monotonic() + config.reset_timeout_s
    latest = previous_sample

    while True:
        if monotonic() >= deadline:
            raise ControlError("index reset to position 0 timed out")
        latest, age = await _next_fresh_sample(
            provider, config, previous=latest, monotonic=monotonic
        )
        position, motor_state = await _read_index_motor(client, slave_id, sdk)
        _report(
            on_telemetry,
            _make_telemetry(
                ControlState.RESETTING,
                latest,
                age,
                position,
                motor_state,
                baseline=None,
            ),
        )
        if position <= INDEX_RESET_POSITION:
            await _set_index_speed(client, slave_id, sdk, 0)
            return latest
        await sleep(config.poll_interval_s)


async def _apply_fast_pressure_action(
    client: Any,
    slave_id: int,
    sdk: Any,
    config: ControlConfig,
    state: ControlState,
    pressure: float,
) -> ControlState:
    """Issue stop/relief commands before motor reads and synchronous output."""

    if state in (ControlState.OBSERVING, ControlState.PRESSING):
        if pressure > config.relief_pressure:
            await _set_index_speed(client, slave_id, sdk, 0)
            await _set_index_speed(client, slave_id, sdk, -config.up_speed)
            return ControlState.RELIEVING
        if pressure > config.stop_pressure:
            await _set_index_speed(client, slave_id, sdk, 0)
            return ControlState.HOLDING
        if state is ControlState.OBSERVING:
            return ControlState.PRESSING
    elif state is ControlState.HOLDING:
        if pressure > config.relief_pressure:
            await _set_index_speed(client, slave_id, sdk, -config.up_speed)
            return ControlState.RELIEVING
    elif state is ControlState.RELIEVING:
        if pressure < config.stop_pressure:
            await _set_index_speed(client, slave_id, sdk, 0)
            return ControlState.HOLDING
    return state


async def run_external_pressure_control(
    client: Any,
    slave_id: int,
    sdk: Any,
    provider: PressureProvider,
    config: ControlConfig,
    *,
    initial_sample: PressureSample,
    on_telemetry: TelemetryCallback | None = None,
    sleep: SleepFunction = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    max_control_samples: int | None = None,
) -> None:
    """Run reset, three-sample baseline, and the external-pressure state machine.

    ``max_control_samples`` is a bounded software-test seam.  Production leaves
    it as ``None``.  The function always attempts a final zero-speed command.
    """

    config.validate()
    initial_sample = _validate_sample(initial_sample)
    if max_control_samples is not None and max_control_samples <= 0:
        raise ValueError("max_control_samples must be greater than zero or None")

    latest = initial_sample
    try:
        # Revalidate the sensor immediately before the first motion command.
        # This sample is intentionally discarded from the post-reset baseline.
        latest, _ = await _next_fresh_sample(
            provider, config, previous=latest, monotonic=monotonic
        )
        latest = await _reset_index_to_zero(
            client,
            slave_id,
            sdk,
            provider,
            config,
            latest,
            on_telemetry=on_telemetry,
            sleep=sleep,
            monotonic=monotonic,
        )

        baseline_values: list[float] = []
        baseline: float | None = None
        for _ in range(BASELINE_VALID_SAMPLE_COUNT):
            latest, age = await _next_fresh_sample(
                provider, config, previous=latest, monotonic=monotonic
            )
            baseline_values.append(latest.pressure_raw)
            if len(baseline_values) == BASELINE_VALID_SAMPLE_COUNT:
                baseline = float(statistics.median(baseline_values))
            position, motor_state = await _read_index_motor(client, slave_id, sdk)
            _report(
                on_telemetry,
                _make_telemetry(
                    ControlState.BASELINING,
                    latest,
                    age,
                    position,
                    motor_state,
                    baseline=baseline,
                ),
            )
            await sleep(config.poll_interval_s)

        assert baseline is not None
        state = ControlState.OBSERVING
        completed = 0
        while True:
            latest, age = await _next_fresh_sample(
                provider, config, previous=latest, monotonic=monotonic
            )
            pressure_control = latest.pressure_raw - baseline
            previous_state = state

            # Threshold action comes before the slower motor transaction and
            # before telemetry output.
            state = await _apply_fast_pressure_action(
                client,
                slave_id,
                sdk,
                config,
                state,
                pressure_control,
            )
            position, motor_state = await _read_index_motor(client, slave_id, sdk)

            # Positive motion is not a fast safety action.  It starts only after
            # finite, non-stalled motor feedback has been checked.
            if (
                previous_state is ControlState.OBSERVING
                and state is ControlState.PRESSING
            ):
                await _set_index_speed(
                    client, slave_id, sdk, config.down_speed
                )

            _report(
                on_telemetry,
                _make_telemetry(
                    state,
                    latest,
                    age,
                    position,
                    motor_state,
                    baseline=baseline,
                    pressure_control=pressure_control,
                ),
            )
            completed += 1
            if max_control_samples is not None and completed >= max_control_samples:
                return
            await sleep(config.poll_interval_s)
    finally:
        try:
            await _set_index_speed(client, slave_id, sdk, 0)
        except Exception:
            pass


def _default_cleanup_error(message: str) -> None:
    print(f"cleanup warning: {message}", file=sys.stderr)


async def run_control_session(
    provider: PressureProvider,
    connect_hand: ConnectHand,
    close_hand: CloseHand,
    sdk: Any,
    config: ControlConfig,
    *,
    expected_hand: str = DEFAULT_EXPECTED_HAND,
    on_telemetry: TelemetryCallback | None = None,
    on_cleanup_error: CleanupErrorCallback = _default_cleanup_error,
    sleep: SleepFunction = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    max_control_samples: int | None = None,
) -> None:
    """High-level injectable session boundary used by production and tests."""

    config.validate()
    if expected_hand not in ("right", "left"):
        raise ValueError("expected_hand must be 'right' or 'left'")

    client: Any = None
    slave_id: int | None = None
    original_unit_mode: Any = None
    try:
        # Model/camera validation happens before Revo2 is connected, so a sensor
        # startup failure cannot send any hand command.
        await provider.start()
        initial_sample, _ = await _next_fresh_sample(
            provider, config, previous=None, monotonic=monotonic
        )

        client, slave_id = await connect_hand()
        info = await client.get_device_info(slave_id)
        hand_label = str(getattr(info, "hand_type", "")).casefold()
        if expected_hand not in hand_label:
            raise ControlError(
                f"hand mismatch: expected {expected_hand}, "
                f"SDK returned {hand_label or info}"
            )

        original_unit_mode = await client.get_finger_unit_mode(slave_id)
        await client.set_finger_unit_mode(
            slave_id, sdk.FingerUnitMode.Normalized
        )
        await run_external_pressure_control(
            client,
            slave_id,
            sdk,
            provider,
            config,
            initial_sample=initial_sample,
            on_telemetry=on_telemetry,
            sleep=sleep,
            monotonic=monotonic,
            max_control_samples=max_control_samples,
        )
    finally:
        if client is not None and slave_id is not None:
            try:
                await _set_index_speed(client, slave_id, sdk, 0)
            except Exception as exc:
                on_cleanup_error(f"final zero-speed command failed: {exc}")

        try:
            await provider.close()
        except Exception as exc:
            on_cleanup_error(f"pressure provider close failed: {exc}")

        if client is not None and slave_id is not None:
            if original_unit_mode is not None:
                try:
                    await client.set_finger_unit_mode(
                        slave_id, original_unit_mode
                    )
                except Exception as exc:
                    on_cleanup_error(f"unit-mode restore failed: {exc}")
            try:
                await close_hand(client)
            except Exception as exc:
                on_cleanup_error(f"Modbus close failed: {exc}")


def format_telemetry(telemetry: Telemetry) -> None:
    control = (
        f"{telemetry.pressure_control:.3f}"
        if math.isfinite(telemetry.pressure_control)
        else "N/A"
    )
    print(
        f"Fx={telemetry.fx:.3f}, Fy={telemetry.fy:.3f}, Fz={telemetry.fz:.3f}, "
        f"resultant={telemetry.pressure_raw:.3f}, "
        f"resultant_after_baseline={control}",
        flush=True,
    )


async def _run_real_hand(args: argparse.Namespace) -> None:
    # Vendor, camera, and deep-learning imports remain out of module import so
    # software tests do not require the hardware stack.
    from revo2_utils import libstark, open_modbus_revo2

    config = ControlConfig(
        stop_pressure=args.stop_pressure,
        relief_pressure=args.relief_pressure,
        down_speed=args.down_speed,
        up_speed=args.up_speed,
        reset_timeout_s=args.reset_timeout,
        poll_interval_s=args.poll_interval,
        max_sample_age_s=args.max_sample_age,
    )
    config.validate()
    provider = CameraModelPressureProvider(camera_index=args.camera_index)

    async def connect_hand() -> tuple[Any, int]:
        return await open_modbus_revo2(
            port_name=args.port,
            quick=not args.full_scan,
        )

    async def close_hand(client: Any) -> None:
        await _maybe_await(libstark.modbus_close(client))

    print(f"model: {ARRAY_SKIN_MODEL_PATH}")
    print(
        f"camera={args.camera_index} DirectShow, capture=1280x720, "
        "resize=640x360, roi=columns[120:480], "
        "pressure=sqrt(Fx^2+Fy^2+Fz^2)"
    )
    print(
        "The sensor is validated before hand connection. After reset, three new "
        "no-contact samples establish the median baseline. Press Ctrl+C to stop."
    )
    await run_control_session(
        provider,
        connect_hand,
        close_hand,
        libstark,
        config,
        expected_hand=args.expected_hand,
        on_telemetry=format_telemetry,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Control the Revo2 index finger from the array_skin visual force model"
        )
    )
    parser.add_argument("--port", default=DEFAULT_PORT_NAME)
    parser.add_argument(
        "--expected-hand",
        choices=("right", "left"),
        default=DEFAULT_EXPECTED_HAND,
    )
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument(
        "--stop-pressure", type=float, default=DEFAULT_STOP_PRESSURE
    )
    parser.add_argument(
        "--relief-pressure", type=float, default=DEFAULT_RELIEF_PRESSURE
    )
    parser.add_argument("--down-speed", type=int, default=DEFAULT_DOWN_SPEED)
    parser.add_argument("--up-speed", type=int, default=DEFAULT_UP_SPEED)
    parser.add_argument(
        "--reset-timeout", type=float, default=DEFAULT_RESET_TIMEOUT_S
    )
    parser.add_argument(
        "--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S
    )
    parser.add_argument(
        "--max-sample-age", type=float, default=DEFAULT_MAX_SAMPLE_AGE_S
    )
    parser.add_argument("--full-scan", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not REAL_MOTION_CONFIRMED:
        parser.error("REAL_MOTION_CONFIRMED is not enabled in this module")

    try:
        asyncio.run(_run_real_hand(args))
    except KeyboardInterrupt:
        print(
            "interrupted by user; final zero-speed and cleanup were attempted",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(f"external pressure control failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
