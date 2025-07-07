import os
import numpy as np
import torch
import torch.distributed as dist
import shutil

import pointops

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu
from collections import OrderedDict

from pointcept.engines.hooks.evaluator import HookBase
from pointcept.engines.hooks.evaluator import HOOKS
from pointcept.engines.hooks.misc import InformationWriter, CheckpointSaver
from pointcept.engines.hooks.misc import IterationTimer
from pointcept.utils.comm import is_main_process


# Custom Hook
@HOOKS.register_module()
class CustomCheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False, skip_key=None):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict
        self.skip_key = skip_key

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )

            # modified from gorilla
            # get model state_dict from checkpoint
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "net" in checkpoint:
                state_dict = checkpoint["net"]
            else:
                state_dict = checkpoint

            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in state_dict.items():
                if not key.startswith("module."):
                    if comm.get_world_size() >= 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                if self.skip_key is not None and self.skip_key in key:
                    continue
                weight[key] = value
                
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class CustomInformationWriter(InformationWriter):
    """
    Compare with the original InformationWriter in Pointcept:
    1. Add the interval for printing the information.
    """
    def __init__(self, interval=1, key=("loss", )):
        super().__init__()
        self.interval = interval
        self.logger_key = key

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            if key in self.logger_key:
                self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )  
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5e}".format(lr=lr)
        if (self.trainer.comm_info["iter"] + 1) % self.interval == 0:
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )


@HOOKS.register_module()
class CustomCheckpointSaver(CheckpointSaver):
    """
    Compare with the original CheckpointSaver in Pointcept:
    1. Add the evaluation interval.
    """
    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_epoch = self.trainer.epoch + 1
                if current_epoch >= self.trainer.cfg.evaluate_interval[0][0]:
                    evaluate_interval = self.trainer.cfg.evaluate_interval[0][1]
                if current_epoch >= self.trainer.cfg.evaluate_interval[1][0]:
                    evaluate_interval = self.trainer.cfg.evaluate_interval[1][1]
            if (self.trainer.cfg.evaluate and 
                current_epoch % evaluate_interval == 0):
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.3f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.3f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": self.trainer.scaler.state_dict()
                    if self.trainer.cfg.enable_amp
                    else None,
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class CustomSemSegEvaluator(HookBase):
    """
    Compare with the original SemSegEvaluator in Pointcept:
    1. Add the evaluation interval for judging whether to evaluate.
    2. Make the print_results more readable.
    """
    def after_epoch(self):
        torch.cuda.empty_cache()
        if self.trainer.cfg.evaluate:
            current_epoch = self.trainer.epoch + 1
            if current_epoch >= self.trainer.cfg.evaluate_interval[0][0]:
                evaluate_interval = self.trainer.cfg.evaluate_interval[0][1]
            if current_epoch >= self.trainer.cfg.evaluate_interval[1][0]:
                evaluate_interval = self.trainer.cfg.evaluate_interval[1][1]
        if (self.trainer.cfg.evaluate and 
            current_epoch % evaluate_interval == 0):
            self.eval()

    def print_results(self, intersection, union, target):
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        
        sep = ""
        col1 = ":"
        lineLen = 40
        self.trainer.logger.info("#" * lineLen)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>10}".format("mIoU") + sep
        line += "{:>10}".format("Acc") + sep
        self.trainer.logger.info(line)
        self.trainer.logger.info("#" * lineLen)

        for i in range(self.trainer.cfg.data.num_classes):
            iou = iou_class[i]
            acc = acc_class[i]
            lable_name = self.trainer.cfg.data.names[i]
            line = "{:<15}".format(lable_name) + sep + col1
            line += sep + "{:>10.4f}".format(iou) + sep
            line += sep + "{:>10.4f}".format(acc) + sep
            self.trainer.logger.info(line)
        self.trainer.logger.info("#" * lineLen)
        self.trainer.logger.info(
            "Overall: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        return m_iou, m_acc, all_acc

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        m_iou, m_acc, all_acc = self.print_results(intersection, union, target)
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver
        torch.cuda.empty_cache()

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class CustomIterationTimer(IterationTimer):
    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Mem R(MA/MR): {res:.0f}" " ({max_alloc:.0f}/{max_res:.0f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    res=torch.cuda.memory_reserved() / (1024 ** 2),
                    max_alloc=torch.cuda.max_memory_allocated() / (1024 ** 2),
                    max_res=torch.cuda.max_memory_reserved() / (1024 ** 2),
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()