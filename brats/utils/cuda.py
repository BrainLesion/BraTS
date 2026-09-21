def normalize_cuda_devices(cuda_devices: str) -> str:
    """Normalize a comma-separated list of CUDA device IDs.

    Splits the input on commas, strips surrounding whitespace from each
    entry, and drops empty entries, so that e.g. ``" 0 , 1 ,"`` becomes
    ``"0,1"``.

    Args:
        cuda_devices (str): Comma-separated list of CUDA device IDs.

    Returns:
        str: Canonical comma-separated list of device IDs.

    Raises:
        ValueError: If the input contains no valid device IDs.
    """
    device_ids = [device for device in cuda_devices.split(",") if device.strip()]
    if not device_ids:
        raise ValueError(
            f"No valid CUDA device IDs in cuda_devices='{cuda_devices}'. "
            f"Expected a comma-separated list of device IDs, e.g. '0' or '0,1'."
        )
    return ",".join(device.strip() for device in device_ids)
