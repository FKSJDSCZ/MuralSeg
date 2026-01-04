from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class KizilDataset(BaseSegDataset):
    """Kizil Cave-Temple Complex dataset.

    In segmentation map annotation for Kizil Cave-Temple Complex dataset, 0 is the background index.
    ``reduce_zero_label`` should be set to False. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=(
            'background',
            'incomplete mural',
            'exposed mud layers',
            'exposed rock layers',
            'protective filling',
            'crack',
            'hole',
        ),
        palette=[
            [119, 11, 32],  # background
            [107, 142, 35],  # incomplete mural
            [0, 60, 100],  # exposed mud layers
            [0, 0, 142],  # exposed rock layers
            [81, 0, 81],  # protective filling
            [250, 170, 30],  # crack
            [150, 120, 90],  # hole
        ]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
