"""
Revo2 Touch Version Dexterous Hand Control Example

This example continuously reads the index-finger normal-force channel from a
Revo2 Touch Version (capacitive sensor) dexterous hand, including:
- Configuration and enabling of tactile sensors
- Continuous acquisition of the index-finger normal-force value
- Sensor status monitoring and exception handling
- Sensor calibration and maintenance operations

Notes:
- This example is only applicable to Revo2 Touch Version (capacitive sensor) hardware
- Tactile sensors are enabled by default on startup
- Do not apply force to the sensor during zero-drift calibration
"""

import asyncio
import sys
from revo2_utils import *


PORT_NAME = "COM11"          # Confirmed right-hand RS-485 port
INDEX_TOUCH_SLOT = 1         # Touch slots: Thumb=0, Index=1, ..., Pinky=4
POLL_INTERVAL_S = 0.05       # 20 Hz read request rate


async def main():
    """
    Main function: initialize the Revo2 Touch Device (capacitive sensor) dexterous hand device and execute the tactile sensor control
    """
    client = None
    try:
        # Connect the confirmed right-hand RS-485 port directly. This avoids
        # the SDK 2.0.3 automatic-port-enumeration failure seen on this PC.
        client, slave_id = await open_modbus_revo2(port_name=PORT_NAME)

        # Verify that the device is the capacitive Revo2 Touch version.
        device_info: libstark.DeviceInfo = await client.get_device_info(slave_id)
        if not device_info.uses_revo2_touch_api():
            raise RuntimeError("This example is only for Revo2 Touch hardware")

        await setup_touch_sensors(client, slave_id)

        logger.info("Continuously reading index-finger pressure; press Ctrl+C to stop")
        await monitor_touch_sensors(client, slave_id)
    finally:
        if client is not None:
            libstark.modbus_close(client)
            logger.info("Modbus client closed")


async def setup_touch_sensors(client, slave_id):
    """
    Configure and enable tactile sensors

    Args:
        client: Modbus client instance
        slave_id: Device ID
    """
    # Enable only the index-finger tactile sensor.
    bits = 1 << INDEX_TOUCH_SLOT
    await client.touch_sensor_setup(slave_id, bits)
    await asyncio.sleep(1)  # Wait for tactile sensors to be ready

    # Verify sensor enabled status
    bits = await client.get_touch_sensor_enabled(slave_id)
    logger.info(f"Touch Sensor Enabled: {(bits & 0x1F):05b}")

    # Get tactile sensor firmware version (can only be obtained after enabling)
    touch_fw_versions = await client.get_touch_sensor_fw_versions(slave_id)
    logger.info(f"Touch Fw Versions: {touch_fw_versions}")


async def monitor_touch_sensors(client, slave_id):
    """
    Continuously read and print the index-finger normal-force channel.

    ``normal_force1`` is printed as the SDK value without assigning a physical
    unit. A conversion to N must be verified for the actual hardware/firmware.

    Args:
        client: Modbus client instance
        slave_id: Device ID
    """
    while True:
        index: libstark.TouchFingerItem = (
            await client.get_single_touch_sensor_status(slave_id, INDEX_TOUCH_SLOT)
        )
        print(
            f"食指法向力/压力通道（SDK原始值）: {index.normal_force1} "
            f"| 传感器状态: {index.status}",
            flush=True,
        )
        await asyncio.sleep(POLL_INTERVAL_S)


async def perform_sensor_maintenance(client, slave_id):
    """
    Perform sensor maintenance operations (optional)

    Args:
        client: Modbus client instance
        slave_id: Device ID
    """
    # Tactile sensor reset
    # Send sensor acquisition channel reset instruction, do not apply force to the finger sensor during execution
    # await client.touch_sensor_reset(slave_id, 0x1f)  # Reset the sensor acquisition channel of the specified finger

    # Tactile sensor parameter calibration
    # When the 3D force value in the idle state is not zero, you can calibrate using this command.
    # This command takes a long time to execute, and data collected during its execution should not be used as reference.
    # It is recommended to ignore sensor data for ten seconds after calibration; do not apply force to the finger sensors during calibration.
    # await client.touch_sensor_calibrate(slave_id, 0x1f)  # Calibrate the sensor data channel for the specified fingers


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("User interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
