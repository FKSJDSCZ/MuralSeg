# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import tempfile
from pathlib import Path
from typing import Any, Tuple, Union

import torch
from mmengine import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.utils import is_tuple_of

from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample

try:
    from mmengine.analysis import ActivationAnalyzer, FlopAnalyzer, parameter_count
    from mmengine.analysis.print_helper import _format_size, complexity_stats_table, complexity_stats_str
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0 to use this script.')


def get_model_complexity_info(
    model: torch.nn.Module,
    input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...], None] = None,
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...], None] = None,
    show_table: bool = True,
    show_arch: bool = True,
    max_depth: int = 3,
):
    """Interface to get the complexity of a model.

    The parameter `inputs` are fed to the forward method of model.
    If `inputs` is not specified, the `input_shape` is required and
    it will be used to construct the dummy input fed to model.
    If the forward of model requires two or more inputs, the `inputs`
    should be a tuple of tensor or the `input_shape` should be a tuple
    of tuple which each element will be constructed into a dumpy input.

    Examples:
        >>> # the forward of model accepts only one input
        >>> input_shape = (3, 224, 224)
        >>> get_model_complexity_info(model, input_shape=input_shape)
        >>> # the forward of model accepts two or more inputs
        >>> input_shape = ((3, 224, 224), (3, 10))
        >>> get_model_complexity_info(model, input_shape=input_shape)

    Args:
        model (nn.Module): The model to analyze.
        input_shape (Union[Tuple[int, ...], Tuple[Tuple[int, ...]], None]):
            The input shape of the model.
            If "inputs" is not specified, the "input_shape" should be set.
            Defaults to None.
        inputs (torch.Tensor, tuple[torch.Tensor, ...] or Tuple[Any, ...],\
            optional]):
            The input tensor(s) of the model. If not given the input tensor
            will be generated automatically with the given input_shape.
            Defaults to None.
        show_table (bool): Whether to show the complexity table.
            Defaults to True.
        show_arch (bool): Whether to show the complexity arch.
            Defaults to True.
        max_depth: Maximum depth of the complexity table.

    Returns:
        dict: The complexity information of the model.
    """
    if input_shape is None and inputs is None:
        raise ValueError('One of "input_shape" and "inputs" should be set.')
    elif input_shape is not None and inputs is not None:
        raise ValueError('"input_shape" and "inputs" cannot be both set.')

    if inputs is None:
        device = next(model.parameters()).device
        if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
            inputs = (torch.randn(1, *input_shape).to(device),)
        elif is_tuple_of(input_shape, tuple) and all(
                [
                    is_tuple_of(one_input_shape, int)
                    for one_input_shape in input_shape  # type: ignore
                ],
        ):  # tuple of tuple of int, construct multiple tensors
            inputs = tuple(
                [
                    torch.randn(1, *one_input_shape).to(device)
                    for one_input_shape in input_shape  # type: ignore
                ],
            )
        else:
            raise ValueError(
                '"input_shape" should be either a `tuple of int` (to construct one input tensor)'
                'or a `tuple of tuple of int` (to construct multiple input tensors).',
            )

    flop_handler = FlopAnalyzer(model, inputs)
    activation_handler = ActivationAnalyzer(model, inputs)

    flops = flop_handler.total()
    activations = activation_handler.total()
    params = parameter_count(model)['']

    flops_str = _format_size(flops)
    activations_str = _format_size(activations)
    params_str = _format_size(params)

    if show_table:
        complexity_table = complexity_stats_table(
            flops=flop_handler,
            max_depth=max_depth,
            activations=activation_handler,
            show_param_shapes=True,
        )
        complexity_table = '\n' + complexity_table
    else:
        complexity_table = ''

    if show_arch:
        complexity_arch = complexity_stats_str(
            flops=flop_handler,
            activations=activation_handler,
        )
        complexity_arch = '\n' + complexity_arch
    else:
        complexity_arch = ''

    return {
        'flops': flops,
        'flops_str': flops_str,
        'activations': activations,
        'activations_str': activations_str,
        'params': params,
        'params_str': params_str,
        'out_table': complexity_table,
        'out_arch': complexity_arch,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs of a segmentor',
    )
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[2048, 1024],
        help='input image size',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.',
    )
    args = parser.parse_args()
    return args


def inference(args: argparse.Namespace, logger: MMLogger) -> dict:
    config_name = Path(args.config)

    if not config_name.exists():
        logger.error(f'Config file {config_name} does not exist')

    cfg: Config = Config.fromfile(config_name)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    cfg.log_level = 'WARN'
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('scope', 'mmseg'))

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')
    result = {}

    model: BaseSegmentor = MODELS.build(cfg.model)
    if hasattr(model, 'auxiliary_head'):
        model.auxiliary_head = None
    if torch.cuda.is_available():
        model.cuda()
    model = revert_sync_batchnorm(model)
    result['ori_shape'] = input_shape[-2:]
    result['pad_shape'] = input_shape[-2:]
    data_batch = {
        'inputs': [torch.rand(input_shape)],
        'data_samples': [SegDataSample(metainfo=result)],
    }
    data = model.data_preprocessor(data_batch)
    model.eval()

    # TODO: Support MaskFormer and Mask2Former
    decode_head = cfg.model.decode_head
    if isinstance(decode_head, dict):
        decode_head = [decode_head]
    for head in decode_head:
        if head['type'] in ['MaskFormerHead', 'Mask2FormerHead']:
            raise NotImplementedError('MaskFormer and Mask2Former are not supported yet.')

    outputs = get_model_complexity_info(
        model,
        input_shape=None,
        inputs=data['inputs'],
        show_table=True,
        show_arch=False,
        max_depth=5,
    )
    for k, v in outputs.items():
        print(f'{k}: {v}')
    result['flops'] = _format_size(outputs['flops'])
    result['params'] = _format_size(outputs['params'])
    result['compute_type'] = 'direct: randomly generate a picture'
    return result


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')

    result = inference(args, logger)
    split_line = '=' * 30
    ori_shape = result['ori_shape']
    pad_shape = result['pad_shape']
    flops = result['flops']
    params = result['params']
    compute_type = result['compute_type']

    if pad_shape != ori_shape:
        print(
            f'{split_line}\nUse size divisor set input shape '
            f'from {ori_shape} to {pad_shape}',
        )
    print(
        f'{split_line}\nCompute type: {compute_type}\n'
        f'Input shape: {pad_shape}\nFlops: {flops}\n'
        f'Params: {params}\n{split_line}',
    )
    print(
        '!!!Please be cautious if you use the results in papers. '
        'You may need to check if all ops are supported and verify '
        'that the flops computation is correct.',
    )


if __name__ == '__main__':
    main()
