import functools
import logging
import os
import os.path as osp
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np
import torch

from mmengine.visualization import LocalVisBackend, WandbVisBackend
from mmengine.config import Config
from mmengine.logging import print_log
from mmseg.registry import VISBACKENDS


def force_init_env(old_func: Callable) -> Any:
    """Those methods decorated by ``force_init_env`` will be forced to call
    ``_init_env`` if the instance has not been fully initiated. This function
    will decorated all the `add_xxx` method and `experiment` method, because
    `VisBackend` is initialized only when used its API.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``_init_env`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `_init_env` method.
        if not hasattr(obj, '_init_env'):
            raise AttributeError(
                f'{type(obj)} does not have _init_env '
                'method.',
            )
        # If instance does not have `_env_initialized` attribute or
        # `_env_initialized` is False, call `_init_env` and set
        # `_env_initialized` to True
        if not getattr(obj, '_env_initialized', False):
            print_log(
                'Attribute `_env_initialized` is not defined in '
                f'{type(obj)} or `{type(obj)}._env_initialized is '
                'False, `_init_env` will be called and '
                f'{type(obj)}._env_initialized will be set to True',
                logger='current',
                level=logging.DEBUG,
            )
            obj._init_env()  # type: ignore
            obj._env_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper


@VISBACKENDS.register_module()
class MyWandbVisBackend(WandbVisBackend):
    """Wandb visualization backend class. This is a improved implementation
    based on mmengine.visualization.WandbVisBackend.

    Examples:
        >>> from mmengine.visualization import WandbVisBackend
        >>> import numpy as np
        >>> wandb_vis_backend = WandbVisBackend()
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> wandb_vis_backend.add_image('img', img)
        >>> wandb_vis_backend.add_scaler('mAP', 0.6)
        >>> wandb_vis_backend.add_scalars({'loss': [1, 2, 3],'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> wandb_vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer.
        init_kwargs (dict, optional): wandb initialization
            input parameters.
            See `wandb.init <https://docs.wandb.ai/ref/python/init>`_ for
            details. Defaults to None.
        define_metric_cfg (dict or list[dict], optional):
            When a dict is set, it is a dict of metrics and summary for
            ``wandb.define_metric``.
            The key is metric and the value is summary.
            When a list is set, each dict should be a valid argument of
            the ``define_metric``.
            For example, ``define_metric_cfg={'coco/bbox_mAP': 'max'}``,
            means the maximum value of ``coco/bbox_mAP`` is logged on wandb UI.
            When ``define_metric_cfg=[dict(name='loss',
            step_metric='epoch')]``,
            the "loss" will be plotted against the epoch.
            See `wandb define_metric <https://docs.wandb.ai/ref/python/
            run#define_metric>`_ for details.
            Defaults to None.
        commit (bool, optional) Save the metrics dict to the wandb server
            and increment the step.  If false `wandb.log` just updates the
            current metrics dict with the row argument and metrics won't be
            saved until `wandb.log` is called with `commit=True`.
            Defaults to True.
        log_code_name (str, optional) The name of code artifact.
            By default, the artifact will be named
            source-$PROJECT_ID-$ENTRYPOINT_RELPATH. See
            `wandb log_code <https://docs.wandb.ai/ref/python/run#log_code>`_
            for details. Defaults to None.
            `New in version 0.3.0.`
        watch_kwargs (optional, dict): Agurments for ``wandb.watch``.
            `New in version 0.4.0.`
    """

    def __init__(
        self,
        save_dir: str,
        init_kwargs: Optional[dict] = None,
        define_metric_cfg: Union[dict, list, None] = None,
        commit: Optional[bool] = True,
        log_code_name: Optional[str] = None,
        watch_kwargs: Optional[dict] = None,
    ):
        super().__init__(
            save_dir,
            init_kwargs,
            define_metric_cfg,
            commit,
            log_code_name,
            watch_kwargs,
        )

    @force_init_env
    def add_image(
        self,
        name: str,
        image: np.ndarray,
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Record the image to wandb.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int, optional): The step number to log. If None,
                then an implicit auto-incrementing step is used.
        """
        image = self._wandb.Image(image)
        self._wandb.log({name: image}, step=step, commit=self._commit)

    @force_init_env
    def add_scalar(
        self,
        name: str,
        value: Union[int, float, torch.Tensor, np.ndarray],
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Record the scalar data to wandb.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int, optional): The step number to log. If None,
                then an implicit auto-incrementing step is used.
        """
        self._wandb.log({name: value}, step=step, commit=self._commit)

    @force_init_env
    def add_scalars(
        self,
        scalar_dict: dict,
        step: Optional[int] = None,
        file_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Record the scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int, optional): The step number to log. If None,
                then an implicit auto-incrementing step is used.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Defaults to None.
        """
        self._wandb.log(scalar_dict, step=step, commit=self._commit)


@VISBACKENDS.register_module()
class MyLocalVisBackend(LocalVisBackend):
    """Local visualization backend class.

    It can write image, config, scalars, etc.
    to the local hard disk. You can get the drawing backend
    through the experiment property for custom drawing.

    Examples:
        >>> from mmengine.visualization import LocalVisBackend
        >>> import numpy as np
        >>> local_vis_backend = LocalVisBackend(save_dir='temp_dir')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> local_vis_backend.add_image('img', img)
        >>> local_vis_backend.add_scalar('mAP', 0.6)
        >>> local_vis_backend.add_scalars({'loss': [1, 2, 3], 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> local_vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer. If it is none, it means no data
            is stored.
        img_save_dir (str): The directory to save images.
            Defaults to 'vis_image'.
        config_save_file (str): The file name to save config.
            Defaults to 'config.py'.
        scalar_save_file (str):  The file name to save scalar values.
            Defaults to 'scalars.json'.
    """

    def __init__(
        self,
        save_dir: str,
        img_save_dir: str = 'vis_image',
        config_save_file: str = 'config.py',
        scalar_save_file: str = 'scalars.json',
    ):
        super().__init__(
            save_dir,
            img_save_dir,
            config_save_file,
            scalar_save_file,
        )

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.array,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        assert image.dtype == np.uint8
        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(self._img_save_dir, exist_ok=True)
        cv2.imwrite(osp.join(self._img_save_dir, name), drawn_image)