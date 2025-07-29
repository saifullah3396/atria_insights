from functools import partial

import torch
from lime.wrappers.scikit_image import SegmentationAlgorithm


def grid_segmenter(images: torch.Tensor, cell_size: int = 16) -> torch.Tensor:
    feature_mask = []
    for image in images:
        # image dimensions are C x H x H
        dim_x, dim_y = image.shape[1] // cell_size, image.shape[2] // cell_size
        mask = (
            torch.arange(dim_x * dim_y, device=images.device)
            .view((dim_x, dim_y))
            .repeat_interleave(cell_size, dim=0)
            .repeat_interleave(cell_size, dim=1)
            .long()
            .unsqueeze(0)
        )
        feature_mask.append(mask)
    return torch.stack(feature_mask)


def _create_segmentation_fn(segmentation_type: str, **kwargs):
    assert segmentation_type in [
        "grid",
        "quickshift",
        "felzenszwalb",
        "slic",
    ], (
        "Segmentation type must be one of 'grid', 'quickshift', 'felzenszwalb', or 'slic'"
    )
    if segmentation_type == "grid":
        return partial(grid_segmenter, **kwargs)
    else:
        return SegmentationAlgorithm(segmentation_type, **kwargs)
