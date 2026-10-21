"""anatoreg: numpy/scipy re-implementation of the antspyx subset AnatoMesh uses (Pyodide-safe, no threads)."""
from .image import ANTsImage, crop_indices, from_numpy, get_mask, image_read, image_write, resample_image, slice_image
from .registration import registration
from .transform import apply_transforms, apply_transforms_to_points, read_transform


def n4_bias_field_correction(*args, **kwargs):
    raise NotImplementedError("anatoreg: n4_bias_field_correction is not implemented yet")


__all__ = ["ANTsImage", "apply_transforms", "apply_transforms_to_points", "crop_indices", "from_numpy", "get_mask",
           "image_read", "image_write", "n4_bias_field_correction", "read_transform", "registration",
           "resample_image", "slice_image"]
