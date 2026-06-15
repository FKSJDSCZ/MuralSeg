import os.path as osp
from collections import deque
from pathlib import Path
from typing import List, Optional, Sequence, Union

from mmengine.hooks import CheckpointHook
from mmengine.fileio import FileClient, get_file_backend
from mmengine.dist import is_main_process
from mmseg.registry import HOOKS


@HOOKS.register_module()
class MyCheckpointHook(CheckpointHook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Defaults to -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Defaults to True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Defaults to True.
        save_param_scheduler (bool): Whether to save param_scheduler state_dict
            in the checkpoint. It is usually used for resuming experiments.
            Defaults to True.
        out_dir (str, Path, Optional): The root directory to save checkpoints.
            If not specified, ``runner.log_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.log_dir``. For example,
            if the input ``our_dir`` is ``./tmp`` and ``runner.log_dir`` is
            ``./work_dir/cur_exp/cur_timestamp``, then the ckpt will be saved in
            ``./tmp/cur_timestamp``. Defaults to None.
        max_keep_ckpts (int): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Defaults to -1, which means unlimited.
        save_last (bool): Whether to force the last checkpoint to be
            saved regardless of interval. Defaults to True.
        save_best (str, List[str], optional): If a metric is specified, it
            would measure the best checkpoint during evaluation. If a list of
            metrics is passed, it would measure a group of best checkpoints
            corresponding to the passed metrics. The information about best
            checkpoint(s) would be saved in ``runner.message_hub`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resuming checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Defaults to None.
        rule (str, List[str], optional): Comparison rule for best score. If
            set to None, it will infer a reasonable rule. Keys such as 'acc',
            'top' .etc will be inferred by 'greater' rule. Keys contain 'loss'
            will be inferred by 'less' rule. If ``save_best`` is a list of
            metrics and ``rule`` is a str, all metrics in ``save_best`` will
            share the comparison rule. If ``save_best`` and ``rule`` are both
            lists, their length must be the same, and metrics in ``save_best``
            will use the corresponding comparison rule in ``rule``. Options
            are 'greater', 'less', None and list which contains 'greater' and
            'less'. Defaults to None.
        greater_keys (List[str], optional): Metric keys that will be
            inferred by 'greater' comparison rule. If ``None``,
            _default_greater_keys will be used. Defaults to None.
        less_keys (List[str], optional): Metric keys that will be
            inferred by 'less' comparison rule. If ``None``, _default_less_keys
            will be used. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
        filename_tmpl (str, optional): String template to indicate checkpoint
            name. If specified, must contain one and only one "{}", which will
            be replaced with ``epoch + 1`` if ``by_epoch=True`` else
            ``iteration + 1``.
            Defaults to None, which means "epoch_{}.pth" or "iter_{}.pth"
            accordingly.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            `New in version 0.2.0.`
        published_keys (str, List[str], optional): If ``save_last`` is ``True``
            or ``save_best`` is not ``None``, it will automatically
            publish model with keys in the list after training.
            Defaults to None.
            `New in version 0.7.1.`
        save_begin (int): Control the epoch number or iteration number
            at which checkpoint saving begins. Defaults to 0, which means
            saving at the beginning.
            `New in version 0.8.3.`

    Examples:
        >>> # Save best based on single metric
        >>> CheckpointHook(interval=2, by_epoch=True, save_best='acc',
        >>>                rule='less')
        >>> # Save best based on multi metrics with the same comparison rule
        >>> CheckpointHook(interval=2, by_epoch=True,
        >>>                save_best=['acc', 'mIoU'], rule='greater')
        >>> # Save best based on multi metrics with different comparison rule
        >>> CheckpointHook(interval=2, by_epoch=True,
        >>>                save_best=['FID', 'IS'], rule=['less', 'greater'])
        >>> # Save best based on single metric and publish model after training
        >>> CheckpointHook(interval=2, by_epoch=True, save_best='acc',
        >>>                rule='less', published_keys=['meta', 'state_dict'])
    """

    def __init__(
        self,
        interval: int = -1,
        by_epoch: bool = True,
        save_optimizer: bool = True,
        save_param_scheduler: bool = True,
        out_dir: Optional[Union[str, Path]] = None,
        max_keep_ckpts: int = -1,
        save_last: bool = True,
        save_best: Union[str, List[str], None] = None,
        rule: Union[str, List[str], None] = None,
        greater_keys: Optional[Sequence[str]] = None,
        less_keys: Optional[Sequence[str]] = None,
        file_client_args: Optional[dict] = None,
        filename_tmpl: Optional[str] = None,
        backend_args: Optional[dict] = None,
        published_keys: Union[str, List[str], None] = None,
        save_begin: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            interval,
            by_epoch,
            save_optimizer,
            save_param_scheduler,
            out_dir,
            max_keep_ckpts,
            save_last,
            save_best,
            rule,
            greater_keys,
            less_keys,
            file_client_args,
            filename_tmpl,
            backend_args,
            published_keys,
            save_begin,
            **kwargs,
        )

    def before_train(self, runner) -> None:
        """Finish all operations, related to checkpoint.

        This function will get the appropriate file client, and the directory
        to save these checkpoints of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.out_dir is None:
            self.out_dir = runner.log_dir

        # If self.file_client_args is None, self.file_client will not
        # used in CheckpointHook. To avoid breaking backward compatibility,
        # it will not be removed util the release of MMEngine1.0
        self.file_client = FileClient.infer_client(
            self.file_client_args,
            self.out_dir,
        )

        if self.file_client_args is None:
            self.file_backend = get_file_backend(
                self.out_dir, backend_args=self.backend_args,
            )
        else:
            self.file_backend = self.file_client

        # if `self.out_dir` is not equal to `runner.log_dir`, it means that `self.out_dir` is set
        # so the final `self.out_dir` is the concatenation of `self.out_dir` and the last level directory of `runner.log_dir`
        if self.out_dir != runner.log_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_backend.join_path(
                self.out_dir, basename,
            )  # type: ignore  # noqa: E501

        runner.logger.info(f'Checkpoints will be saved to {self.out_dir}.')

        if self.save_best is not None:
            if len(self.key_indicators) == 1:
                if 'best_ckpt' not in runner.message_hub.runtime_info:
                    self.best_ckpt_path = None
                else:
                    self.best_ckpt_path = runner.message_hub.get_info(
                        'best_ckpt',
                    )
            else:
                for key_indicator in self.key_indicators:
                    best_ckpt_name = f'best_ckpt_{key_indicator}'
                    if best_ckpt_name not in runner.message_hub.runtime_info:
                        self.best_ckpt_path_dict[key_indicator] = None
                    else:
                        self.best_ckpt_path_dict[
                            key_indicator] = runner.message_hub.get_info(
                            best_ckpt_name,
                        )

        if self.max_keep_ckpts > 0:
            keep_ckpt_ids = []
            if 'keep_ckpt_ids' in runner.message_hub.runtime_info:
                keep_ckpt_ids = runner.message_hub.get_info('keep_ckpt_ids')

                while len(keep_ckpt_ids) > self.max_keep_ckpts:
                    step = keep_ckpt_ids.pop(0)
                    if is_main_process():
                        path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(step),
                        )
                        if self.file_backend.isfile(path):
                            self.file_backend.remove(path)
                        elif self.file_backend.isdir(path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(path)

            self.keep_ckpt_ids: deque = deque(
                keep_ckpt_ids,
                self.max_keep_ckpts,
            )

    def _save_checkpoint_with_step(self, runner, step, meta):
        # remove other checkpoints before save checkpoint to make the
        # self.keep_ckpt_ids are saved as expected
        if self.max_keep_ckpts > 0:
            # _save_checkpoint and _save_best_checkpoint may call this
            # _save_checkpoint_with_step in one epoch
            if len(self.keep_ckpt_ids) > 0 and self.keep_ckpt_ids[-1] == step:
                pass
            else:
                if len(self.keep_ckpt_ids) == self.max_keep_ckpts:
                    _step = self.keep_ckpt_ids.popleft()
                    if is_main_process():
                        ckpt_path = self.file_backend.join_path(
                            self.out_dir, self.filename_tmpl.format(_step),
                        )

                        if self.file_backend.isfile(ckpt_path):
                            self.file_backend.remove(ckpt_path)
                        elif self.file_backend.isdir(ckpt_path):
                            # checkpoints saved by deepspeed are directories
                            self.file_backend.rmtree(ckpt_path)

                self.keep_ckpt_ids.append(step)
                runner.message_hub.update_info(
                    'keep_ckpt_ids',
                    list(self.keep_ckpt_ids),
                )

        ckpt_filename = self.filename_tmpl.format(step)
        self.last_ckpt = self.file_backend.join_path(
            self.out_dir,
            ckpt_filename,
        )
        runner.message_hub.update_info('last_ckpt', self.last_ckpt)

        runner.save_checkpoint(
            self.out_dir,
            ckpt_filename,
            self.file_client_args,
            save_optimizer=self.save_optimizer,
            save_param_scheduler=self.save_param_scheduler,
            meta=meta,
            by_epoch=self.by_epoch,
            backend_args=self.backend_args,
            **self.args,
        )

        # Model parallel-like training should involve pulling sharded states
        # from all ranks, but skip the following procedure.
        if not is_main_process():
            return

        save_file = osp.join(self.out_dir, 'last_checkpoint')
        with open(save_file, 'w') as f:
            f.write(self.last_ckpt)  # type: ignore
