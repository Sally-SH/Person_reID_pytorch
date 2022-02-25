from __future__ import absolute_import

from .pcb_rpp import pcb
from .alignedReID import alignedReID
from .mgn import mgn

__model_factory = {
    # image classification models
    'pcb': pcb,
    'alignedReID': alignedReID,
    'mgn': mgn
}


def show_avai_models():
    """Displays available models.
    """
    print(list(__model_factory.keys()))


def build_model(
    name, num_classes, loss='softmax', pretrained=True, use_gpu=True, **kwargs):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    if name == 'pcb':
        return pcb(num_classes=num_classes,pretrained=pretrained,**kwargs)
    elif name == 'alignedReID':
        return alignedReID(num_classes=num_classes, pretrained=pretrained)
    elif name == 'mgn':
        return mgn(num_classes)

    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )