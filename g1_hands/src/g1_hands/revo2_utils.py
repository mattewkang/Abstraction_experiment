"""
Utility helpers for the BrainCo Revo2 dexterous hand.

Wraps `bc_stark_sdk` so callers can:
* auto-detect a Modbus connection on a given USB port and open the client,
* enumerate available ports from the SDK.

Failures raise `RuntimeError` instead of calling `sys.exit`; the hand node
uses its own jittered retry wrapper on top of `open_modbus_revo2`.
"""

import json

from bc_stark_sdk import main_mod

libstark = main_mod


async def open_modbus_revo2(port_name=None):
    """
    Auto-detect and open a Revo2 Modbus connection.

    Returns:
        tuple: (client, slave_id).

    Raises:
        RuntimeError: SDK auto-detect failed or the detected protocol is
            not Modbus.
    """
    quick = True

    try:
        protocol, port_name, baudrate, slave_id = await libstark.auto_detect_modbus_revo2(
            port_name, quick
        )
        assert (
            protocol == libstark.StarkProtocolType.Modbus
        ), "Only Modbus protocol is supported for Revo2"
    except Exception as e:
        raise RuntimeError(
            f"Failed to auto-detect Revo2 Modbus on port={port_name}: {e}"
        ) from e

    client: libstark.DeviceContext = await libstark.modbus_open(port_name, baudrate)
    await client.get_device_info(slave_id)
    return client, slave_id


def get_stark_port_name():
    """
    Return the first available Stark device port reported by the SDK, or
    `None` when no port is enumerated.
    """
    ports = libstark.list_available_ports()
    ports_json = json.loads(ports.decode("utf-8"))
    if not ports_json:
        return None
    return ports_json[0]["port_name"]
