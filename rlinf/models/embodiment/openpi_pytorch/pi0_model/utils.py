import torch


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    # if dtype_str == "mp_bfloat16":
    #     assert False
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "mp_bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_str]
