from __future__ import print_function, absolute_import

from .cuhk03 import CUHK03
from .market1501 import Market1501
from .mars import Mars
from .dukemtmcreid import DukeMTMCreID

__image_datasets = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID
}

__video_datasets = {
    'mars': Mars
}


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name](**kwargs)


def init_video_dataset(name, **kwargs):
    """Initializes a video dataset."""
    avai_datasets = list(__video_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __video_datasets[name](**kwargs)