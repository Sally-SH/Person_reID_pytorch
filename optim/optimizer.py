from __future__ import print_function, absolute_import
import torch

OPTIMS = ['adam', 'sgd', 'sgd_rpp']

def build_optimizer(
    model,
    optim='adam',
    lr=0.0003,
    weight_decay=5e-04,
    momentum=0.9,
    sgd_dampening=0,
    sgd_nesterov=False,
    adam_beta1=0.9,
    adam_beta2=0.99,
):
    """A function wrapper for building an optimizer.
    Args:
        model (nn.Module): model.
        optim (str, optional): optimizer. Default is "adam".
        lr (float, optional): learning rate. Default is 0.0003.
        weight_decay (float, optional): weight decay (L2 penalty). Default is 5e-04.
        momentum (float, optional): momentum factor in sgd. Default is 0.9,
        sgd_dampening (float, optional): dampening for momentum. Default is 0.
        sgd_nesterov (bool, optional): enables Nesterov momentum. Default is False.
        adam_beta1 (float, optional): beta-1 value in adam. Default is 0.9.
        adam_beta2 (float, optional): beta-2 value in adam. Default is 0.99,
    """

    if optim not in OPTIMS:
        raise ValueError(
            'Unsupported optim: {}. Must be one of {}'.format(
                optim, OPTIMS
            )
        )

    param_groups = model.parameters()

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )
    elif optim == 'sgd_rpp':
        optimizer = torch.optim.SGD(
            model.parts_avgpool.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    return optimizer