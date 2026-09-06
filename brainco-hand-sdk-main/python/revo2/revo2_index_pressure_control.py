"""Revo2 电容触觉版：食指持续下压、停压与超压撤回示例。

启动时先将食指复位到位置 0，再用连续有效样本建立无接触基线，然后持续读取食指
``normal_force1`` 原始值：

1. 以“原始值减启动基线”作为控制值；
2. 控制值不高于停止阈值时，按正速度驱动食指下压；
3. 控制值首次严格大于停止阈值时，停止下压并保持；
4. 控制值严格大于上提阈值时，先停下压再以反向速度缓慢上提；
5. 上提过程中控制值严格小于停止阈值时，停止上提并保持；
6. 触觉状态码及其低字节仅记录并输出，不参与基线或运动判断。

重要：阈值是 SDK 原始通道值相对启动基线的差值，不是 N 或 Pa。只有安装姿态合适
时，食指屈曲才对应空间中的“下压”。本程序不是硬件急停，也没有经过当前真机验证。
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


DEFAULT_STOP_PRESSURE = 70.0
DEFAULT_RELIEF_PRESSURE = 100.0
DEFAULT_PORT_NAME = "COM11"
DEFAULT_DOWN_SPEED = 100
DEFAULT_UP_SPEED = 50
DEFAULT_RESET_TIMEOUT_S = 5.0
DEFAULT_POLL_INTERVAL_S = 0.05
REAL_MOTION_CONFIRMED = True
INDEX_TOUCH_SLOT = 1
INDEX_TOUCH_ENABLE_BIT = 0x02
INDEX_RESET_POSITION = 0.0
TACTILE_ERROR_MASK = 0x00FF
BASELINE_VALID_SAMPLE_COUNT = 3
FINGER_ID_TO_MOTOR_SLOT_OFFSET = 1
SDK_SPEED_MAX = 1000


class ControlError(RuntimeError):
    """Raised after the controller has attempted to stop the index finger."""


class ControlState(str, Enum):
    RESETTING = "RESETTING"
    BASELINING = "BASELINING"
    OBSERVING = "OBSERVING"
    PRESSING = "PRESSING"
    HOLDING = "HOLDING"
    RELIEVING = "RELIEVING"


@dataclass(frozen=True)
class ControlConfig:
    """Parameters that must be checked against the real hand before motion."""

    stop_pressure: float
    relief_pressure: float
    down_speed: int
    up_speed: int
    reset_timeout_s: float
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S

    def validate(self) -> None:
        if not math.isfinite(self.stop_pressure) or self.stop_pressure < 0:
            raise ValueError("stop_pressure 必须是非负有限数值")
        if not math.isfinite(self.relief_pressure):
            raise ValueError("relief_pressure 必须是有限数值")
        if self.relief_pressure <= self.stop_pressure:
            raise ValueError("relief_pressure 必须大于 stop_pressure")
        if not 1 <= self.down_speed <= SDK_SPEED_MAX:
            raise ValueError("down_speed 必须在 1..1000 内")
        if not 1 <= self.up_speed <= SDK_SPEED_MAX:
            raise ValueError("up_speed 必须在 1..1000 内；程序会自动发送负号")
        if not math.isfinite(self.reset_timeout_s) or self.reset_timeout_s <= 0:
            raise ValueError("reset_timeout_s 必须大于 0")
        if not math.isfinite(self.poll_interval_s) or self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s 必须大于 0")


@dataclass(frozen=True)
class Telemetry:
    state: ControlState
    pressure_raw: float
    position: float
    motor_state: str
    touch_status: int | None = None
    touch_error_bits: int | None = None
    pressure_control: float = math.nan
    pressure_baseline: float | None = None


@dataclass(frozen=True)
class TouchSample:
    pressure_raw: float
    touch_status: int
    touch_error_bits: int


@dataclass
class ControlRuntime:
    state: ControlState = ControlState.BASELINING
    baseline_samples: list[float] = field(default_factory=list)
    pressure_baseline: float | None = None


TelemetryCallback = Callable[[Telemetry], None]
SleepFunction = Callable[[float], Awaitable[None]]


async def _read_index_motor(
    client: Any,
    slave_id: int,
    sdk: Any,
) -> tuple[float, str]:
    motor = await client.get_motor_status(slave_id)
    # SDK FingerId values are one-based (Index=3), while motor arrays are
    # zero-based [Thumb, ThumbAux, Index, Middle, Ring, Pinky].
    index = int(sdk.FingerId.Index) - FINGER_ID_TO_MOTOR_SLOT_OFFSET
    if index < 0 or index >= len(motor.positions) or index >= len(motor.states):
        raise ControlError(
            "电机状态数组不包含食指数据: "
            f"finger_id={int(sdk.FingerId.Index)}, "
            f"positions={len(motor.positions)}, states={len(motor.states)}"
        )
    position = float(motor.positions[index])
    motor_state = str(motor.states[index])
    if not math.isfinite(position):
        raise ControlError(f"食指位置读数不是有限数值: {position!r}")
    if "stall" in motor_state.casefold():
        raise ControlError(f"食指电机检测到堵转: {motor_state}")
    return position, motor_state


async def _read_index_touch(
    client: Any,
    slave_id: int,
) -> TouchSample:
    touch = await client.get_single_touch_sensor_status(slave_id, INDEX_TOUCH_SLOT)
    touch_status = int(touch.status) & 0xFFFF
    touch_error_bits = touch_status & TACTILE_ERROR_MASK

    pressure = float(touch.normal_force1)
    if not math.isfinite(pressure):
        raise ControlError(f"食指压力读数不是有限数值: {pressure!r}")

    return TouchSample(
        pressure_raw=pressure,
        touch_status=touch_status,
        touch_error_bits=touch_error_bits,
    )


async def _set_index_speed(client: Any, slave_id: int, sdk: Any, speed: int) -> None:
    await client.set_finger_speed(slave_id, sdk.FingerId.Index, speed)


def _report(callback: TelemetryCallback | None, telemetry: Telemetry) -> None:
    if callback is not None:
        callback(telemetry)


def _check_reset_deadline(deadline: float, monotonic: Callable[[], float]) -> None:
    if monotonic() >= deadline:
        raise ControlError("食指复位到 0 位置超时")


async def _reset_index_to_zero(
    client: Any,
    slave_id: int,
    sdk: Any,
    config: ControlConfig,
    *,
    on_telemetry: TelemetryCallback | None = None,
    sleep: SleepFunction = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    await client.set_finger_position_with_speed(
        slave_id,
        sdk.FingerId.Index,
        int(INDEX_RESET_POSITION),
        config.up_speed,
    )
    deadline = monotonic() + config.reset_timeout_s

    while True:
        await sleep(config.poll_interval_s)
        _check_reset_deadline(deadline, monotonic)
        position, motor_state = await _read_index_motor(client, slave_id, sdk)
        telemetry = Telemetry(
            ControlState.RESETTING,
            math.nan,
            position,
            motor_state,
        )
        _report(on_telemetry, telemetry)
        if telemetry.position <= INDEX_RESET_POSITION:
            await _set_index_speed(client, slave_id, sdk, 0)
            return


async def _react_to_pressure(
    client: Any,
    slave_id: int,
    sdk: Any,
    config: ControlConfig,
    state: ControlState,
    pressure: float,
) -> ControlState:
    """Apply threshold actions before slower motor reads or telemetry output."""

    if state in (ControlState.OBSERVING, ControlState.PRESSING):
        if pressure > config.relief_pressure:
            # Stop closing before reversing into the opening direction.
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


async def _control_one_sample(
    client: Any,
    slave_id: int,
    sdk: Any,
    config: ControlConfig,
    runtime: ControlRuntime,
    on_telemetry: TelemetryCallback | None,
) -> ControlState:
    sample_state = runtime.state
    touch = await _read_index_touch(client, slave_id)
    pressure_control = math.nan

    if runtime.pressure_baseline is None:
        # Position 0 is the declared no-contact startup pose. Use a small odd
        # sample set and its median so one valid-but-noisy reading cannot become
        # the baseline by itself. No motor-speed command is issued here.
        runtime.baseline_samples.append(touch.pressure_raw)
        if len(runtime.baseline_samples) >= BASELINE_VALID_SAMPLE_COUNT:
            runtime.pressure_baseline = float(
                statistics.median(runtime.baseline_samples)
            )
            runtime.state = ControlState.OBSERVING
    else:
        pressure_control = touch.pressure_raw - runtime.pressure_baseline

        # Pressure safety/relief commands are deliberately issued before the
        # motor status transaction and synchronous telemetry callback. Both can
        # block on a real serial connection or slow output stream.
        runtime.state = await _react_to_pressure(
            client,
            slave_id,
            sdk,
            config,
            sample_state,
            pressure_control,
        )

    position, motor_state = await _read_index_motor(client, slave_id, sdk)

    # Starting downward motion is not an urgent safety action. Validate motor
    # feedback first, then start pressing, still before potentially slow output.
    if (
        sample_state is ControlState.OBSERVING
        and runtime.state is ControlState.PRESSING
    ):
        await _set_index_speed(client, slave_id, sdk, config.down_speed)

    _report(
        on_telemetry,
        Telemetry(
            state=sample_state,
            pressure_raw=touch.pressure_raw,
            position=position,
            motor_state=motor_state,
            touch_status=touch.touch_status,
            touch_error_bits=touch.touch_error_bits,
            pressure_control=pressure_control,
            pressure_baseline=runtime.pressure_baseline,
        ),
    )
    return runtime.state


async def _wait_for_next_poll(
    deadline: float,
    interval: float,
    sleep: SleepFunction,
    monotonic: Callable[[], float],
) -> float:
    """Guarantee a full quiet interval between consecutive control cycles."""

    del deadline
    await sleep(interval)
    return monotonic()


async def run_pressure_monitor(
    client: Any,
    slave_id: int,
    sdk: Any,
    config: ControlConfig,
    *,
    on_telemetry: TelemetryCallback | None = None,
    sleep: SleepFunction = asyncio.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    max_monitor_samples: int | None = None,
) -> None:
    """Reset the index finger, then continuously monitor and control pressure.

    ``up_speed`` is configured as a positive magnitude. The function sends it as
    a negative SDK speed because the vendor example defines negative speed as
    opening and positive speed as closing. ``max_monitor_samples`` is only a
    bounded-run seam for software tests; real operation leaves it as ``None``.
    """

    config.validate()
    if max_monitor_samples is not None and max_monitor_samples <= 0:
        raise ValueError("max_monitor_samples 必须大于 0 或为 None")

    try:
        await _reset_index_to_zero(
            client,
            slave_id,
            sdk,
            config,
            on_telemetry=on_telemetry,
            sleep=sleep,
            monotonic=monotonic,
        )
        runtime = ControlRuntime()
        monitor_samples = 0
        next_poll_deadline = monotonic()

        while True:
            await _control_one_sample(
                client,
                slave_id,
                sdk,
                config,
                runtime,
                on_telemetry,
            )
            monitor_samples += 1
            if (
                max_monitor_samples is not None
                and monitor_samples >= max_monitor_samples
            ):
                return

            next_poll_deadline = await _wait_for_next_poll(
                next_poll_deadline,
                config.poll_interval_s,
                sleep,
                monotonic,
            )
    finally:
        # Best-effort fail-closed cleanup. This software zero-speed command is not
        # a certified hardware emergency stop.
        try:
            await _set_index_speed(client, slave_id, sdk, 0)
        except Exception:
            pass


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _format_telemetry(telemetry: Telemetry) -> None:
    pressure = (
        f"{telemetry.pressure_raw:.3f}"
        if math.isfinite(telemetry.pressure_raw)
        else "N/A"
    )
    touch_status = (
        f"0x{telemetry.touch_status:04X}"
        if telemetry.touch_status is not None
        else "N/A"
    )
    touch_error_bits = (
        f"0x{telemetry.touch_error_bits:02X}"
        if telemetry.touch_error_bits is not None
        else "N/A"
    )
    pressure_control = (
        f"{telemetry.pressure_control:.3f}"
        if math.isfinite(telemetry.pressure_control)
        else "N/A"
    )
    pressure_baseline = (
        f"{telemetry.pressure_baseline:.3f}"
        if telemetry.pressure_baseline is not None
        else "N/A"
    )
    print(
        f"[{telemetry.state.value}] "
        f"pressure_raw={pressure}, ",
        # f"pressure_control={pressure_control}, "
        # f"pressure_baseline={pressure_baseline}, "
        # f"position={telemetry.position:.1f}, "
        # f"motor_state={telemetry.motor_state}, "
        # f"touch_status={touch_status}, "
        # f"touch_error_bits={touch_error_bits}",
        flush=True,
    )


async def _run_real_hand(args: argparse.Namespace) -> None:
    # Keep vendor imports inside the real-hardware entry point so the control
    # algorithm can be unit-tested without installing or connecting the SDK.
    from revo2_utils import libstark, open_modbus_revo2

    config = ControlConfig(
        stop_pressure=args.stop_pressure,
        relief_pressure=args.relief_pressure,
        down_speed=args.down_speed,
        up_speed=args.up_speed,
        reset_timeout_s=args.reset_timeout,
        poll_interval_s=args.poll_interval,
    )
    # Reject unsafe/incomplete arguments before opening the real serial device.
    config.validate()

    client = None
    slave_id = None
    original_unit_mode = None
    original_touch_bits = None

    try:
        client, slave_id = await open_modbus_revo2(
            port_name=args.port,
            quick=not args.full_scan,
        )
        info = await client.get_device_info(slave_id)

        uses_touch_api = getattr(info, "uses_revo2_touch_api", None)
        if not callable(uses_touch_api) or not uses_touch_api():
            raise ControlError(f"设备不是 Revo2 Touch 触觉版: {info}")

        uses_pressure_api = getattr(client, "uses_pressure_touch_api", None)
        if not callable(uses_pressure_api):
            raise ControlError("SDK 无法确认触觉类型；为避免误用，禁止开始运动")
        is_pressure_touch = bool(
            await _maybe_await(uses_pressure_api(slave_id))
        )
        if is_pressure_touch:
            raise ControlError("检测到压力型触觉硬件，本程序只适用于电容式触觉版")

        hand_label = str(getattr(info, "hand_type", "")).casefold()
        if args.expected_hand not in hand_label:
            raise ControlError(
                f"左右手不匹配：期望 {args.expected_hand}，SDK 返回 {hand_label or info}"
            )

        original_unit_mode = await client.get_finger_unit_mode(slave_id)
        await client.set_finger_unit_mode(
            slave_id, libstark.FingerUnitMode.Normalized
        )

        original_touch_bits = int(
            await client.get_touch_sensor_enabled(slave_id)
        )
        await client.touch_sensor_setup(
            slave_id, original_touch_bits | INDEX_TOUCH_ENABLE_BIT
        )
        await asyncio.sleep(1.0)
        enabled_bits = int(await client.get_touch_sensor_enabled(slave_id))
        if not enabled_bits & INDEX_TOUCH_ENABLE_BIT:
            raise ControlError("食指电容触觉传感器未成功启用")

        touch_versions = await client.get_touch_sensor_fw_versions(slave_id)
        print(f"设备: {getattr(info, 'description', info)}")
        print(f"设备 ID: {slave_id}; 食指触觉固件: {touch_versions}")
        print(
            "将先把食指复位到位置 0，并用连续有效样本建立无接触基线；"
            "随后首次基线校正值 "
            f">{config.stop_pressure:g} 时停止下压，校正值 "
            f">{config.relief_pressure:g} 时上提，校正值 "
            f"<{config.stop_pressure:g} 时停止上提。"
        )
        print(
            "程序将持续监测并输出 SDK normal_force1 原始值；按 Ctrl+C 结束。",
            flush=True,
        )
        print(
            "触觉 status 及其低字节仅记录，不参与运动判断；"
            "压力/电机/SDK 调用异常仍会停机。",
            flush=True,
        )

        await run_pressure_monitor(
            client,
            slave_id,
            libstark,
            config,
            on_telemetry=_format_telemetry,
        )
    finally:
        if client is not None and slave_id is not None:
            try:
                await client.set_finger_speed(
                    slave_id, libstark.FingerId.Index, 0
                )
            except Exception as exc:
                print(f"警告：最终零速命令失败: {exc}", file=sys.stderr)
            if original_touch_bits is not None:
                try:
                    await client.touch_sensor_setup(slave_id, original_touch_bits)
                except Exception as exc:
                    print(f"警告：恢复触觉使能位失败: {exc}", file=sys.stderr)
            if original_unit_mode is not None:
                try:
                    await client.set_finger_unit_mode(slave_id, original_unit_mode)
                except Exception as exc:
                    print(f"警告：恢复单位模式失败: {exc}", file=sys.stderr)
            try:
                await _maybe_await(libstark.modbus_close(client))
            except Exception as exc:
                print(f"警告：关闭 Modbus 连接失败: {exc}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Revo2 电容触觉版食指持续压力监测与下压/上提控制"
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT_NAME,
        help="Windows 串口，默认 COM11",
    )
    parser.add_argument(
        "--expected-hand",
        choices=("right", "left"),
        default="right",
        help="期望连接的左右手，默认 right",
    )
    parser.add_argument(
        "--stop-pressure",
        type=float,
        default=DEFAULT_STOP_PRESSURE,
        help=(
            "首次停止下压及停止上提的压力边界，"
            f"默认 {DEFAULT_STOP_PRESSURE:g}"
        ),
    )
    parser.add_argument(
        "--relief-pressure",
        type=float,
        default=DEFAULT_RELIEF_PRESSURE,
        help=f"触发上提的压力边界，默认 {DEFAULT_RELIEF_PRESSURE:g}",
    )
    parser.add_argument(
        "--down-speed",
        type=int,
        default=DEFAULT_DOWN_SPEED,
        help=(
            "下压速度的正值幅度，"
            f"默认 {DEFAULT_DOWN_SPEED}，范围 1..1000"
        ),
    )
    parser.add_argument(
        "--up-speed",
        type=int,
        default=DEFAULT_UP_SPEED,
        help=(
            "上提速度的正值幅度，"
            f"默认 {DEFAULT_UP_SPEED}；程序实际发送 -{DEFAULT_UP_SPEED}，"
            "范围 1..1000"
        ),
    )
    parser.add_argument(
        "--reset-timeout",
        type=float,
        default=DEFAULT_RESET_TIMEOUT_S,
        help="启动时食指复位到位置 0 的最长秒数，默认 5",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help="触觉/电机轮询间隔秒数，默认 0.05（约 20 Hz）",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="使用完整设备 ID 扫描；通常比默认快速扫描慢很多",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not REAL_MOTION_CONFIRMED:
        parser.error("代码中的 REAL_MOTION_CONFIRMED 未启用")

    try:
        asyncio.run(_run_real_hand(args))
    except KeyboardInterrupt:
        print("用户中断；程序已尝试发送食指零速命令。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"控制失败: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
