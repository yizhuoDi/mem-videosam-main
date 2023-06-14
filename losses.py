from functools import partial
from typing import Callable, Optional, Union
from ocl.matching import CPUHungarianMatcher
import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
from ocl import base, consistency, path_defaults, scheduling
from ocl.utils import RoutableMixin
from typing import Any
from scipy.optimize import linear_sum_assignment
from ocl.base import Instances
from torchvision.ops import generalized_box_iou
from ocl.utils import box_cxcywh_to_xyxy


def _constant_weight(weight: float, global_step: int):
    return weight


class ReconstructionLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_type: str,
        weight: Union[Callable, float] = 1.0,
        normalize_target: bool = False,
        input_path: Optional[str] = None,
        target_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {"input": input_path, "target": target_path, "global_step": path_defaults.GLOBAL_STEP},
        )
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "mse_sum":
            # Used for slot_attention and video slot attention.
            self.loss_fn = (
                lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="sum") / x1.shape[0]
            )
        elif loss_type == "l1":
            self.loss_name = "l1_loss"
            self.loss_fn = nn.functional.l1_loss
        elif loss_type == "cosine":
            self.loss_name = "cosine_loss"
            self.loss_fn = lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss {loss_type}. Valid choices are (mse, l1, cosine).")
        # If weight is callable use it to determine scheduling otherwise use constant value.
        self.weight = weight if callable(weight) else partial(_constant_weight, weight)
        self.normalize_target = normalize_target

    @RoutableMixin.route
    def forward(self, input: torch.Tensor, target: torch.Tensor, global_step: int):
        target = target.detach()
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = self.loss_fn(input, target)
        weight = self.weight(global_step)
        return weight * loss


class LatentDupplicateSuppressionLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        weight: Union[float, scheduling.HPSchedulerT],
        eps: float = 1e-08,
        grouping_path: Optional[str] = "perceptual_grouping",
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self, {"grouping": grouping_path, "global_step": path_defaults.GLOBAL_STEP}
        )
        self.weight = weight
        self.similarity = nn.CosineSimilarity(dim=-1, eps=eps)

    @RoutableMixin.route
    def forward(self, grouping: base.PerceptualGroupingOutput, global_step: int):
        if grouping.dim() == 4:
            # Build large tensor of reconstructed video.
            # objects = grouping.objects
            objects = grouping
            bs, n_frames, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, :, off_diag_indices[0], :], objects[:, :, off_diag_indices[1], :]
                )
                ** 2
            )

            # if grouping.is_empty is not None:
            #     p_not_empty = 1.0 - grouping.is_empty
            #     # Assume that the probability of of individual objects being present is independent,
            #     # thus the probability of both being present is the product of the individual
            #     # probabilities.
            #     p_pair_present = (
            #         p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
            #     )
            #     # Use average expected penalty as loss for each frame.
            #     losses = (sq_similarities * p_pair_present) / torch.sum(
            #         p_pair_present, dim=-1, keepdim=True
            #     )
            # else:
            losses = sq_similarities.mean(dim=-1)

            weight = self.weight(global_step) if callable(self.weight) else self.weight

            return weight * losses.sum() / (bs * n_frames)
        elif grouping.dim() == 3:
            # Build large tensor of reconstructed image.
            objects = grouping
            bs, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, off_diag_indices[0], :], objects[:, off_diag_indices[1], :]
                )
                ** 2
            )

            if grouping.is_empty is not None:
                p_not_empty = 1.0 - grouping.is_empty
                # Assume that the probability of of individual objects being present is independent,
                # thus the probability of both being present is the product of the individual
                # probabilities.
                p_pair_present = (
                    p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
                )
                # Use average expected penalty as loss for each frame.
                losses = (sq_similarities * p_pair_present) / torch.sum(
                    p_pair_present, dim=-1, keepdim=True
                )
            else:
                losses = sq_similarities.mean(dim=-1)

            weight = self.weight(global_step) if callable(self.weight) else self.weight
            return weight * losses.sum() / bs
        else:
            raise ValueError("Incompatible input format.")


class ConsistencyLoss(nn.Module, RoutableMixin):
    """Task that returns the previously extracted objects.

    Intended to make the object representations accessible to downstream functions, e.g. metrics.
    """

    def __init__(
        self,
        matcher: consistency.HungarianMatcher,
        loss_type: str = "CE",
        loss_weight: float = 0.25,
        mask_path: Optional[str] = None,
        mask_target_path: Optional[str] = None,
        params_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "mask": mask_path,
                "mask_target": mask_target_path,
                "cropping_params": params_path,
                "global_step": path_defaults.GLOBAL_STEP,
            },
        )
        self.matcher = matcher
        if loss_type == "CE":
            self.loss_name = "masks_consistency_CE"
            self.weight = (
                loss_weight if callable(loss_weight) else partial(_constant_weight, loss_weight)
            )
            self.loss_fn = nn.CrossEntropyLoss()

    @RoutableMixin.route
    def forward(
        self,
        mask: torch.Tensor,
        mask_target: torch.Tensor,
        cropping_params: torch.Tensor,
        global_step: int,
    ):
        _, n_objects, size, _ = mask.shape
        mask_one_hot = self._to_binary_mask(mask)
        mask_target = self.crop_views(mask_target, cropping_params, size)
        mask_target_one_hot = self._to_binary_mask(mask_target)
        match = self.matcher(mask_one_hot, mask_target_one_hot)
        matched_mask = torch.stack([mask[match[i, 1]] for i, mask in enumerate(mask)])
        assert matched_mask.shape == mask.shape
        assert mask_target.shape == mask.shape
        flattened_matched_mask = matched_mask.permute(0, 2, 3, 1).reshape(-1, n_objects)
        flattened_mask_target = mask_target.permute(0, 2, 3, 1).reshape(-1, n_objects)
        weight = self.weight(global_step) if callable(self.weight) else self.weight
        return weight * self.loss_fn(flattened_matched_mask, flattened_mask_target)

    @staticmethod
    def _to_binary_mask(masks: torch.Tensor):
        _, n_objects, _, _ = masks.shape
        m_lables = masks.argmax(dim=1)
        mask_one_hot = torch.nn.functional.one_hot(m_lables, n_objects)
        return mask_one_hot.permute(0, 3, 1, 2)

    def crop_views(self, view: torch.Tensor, param: torch.Tensor, size: int):
        return torch.cat([self.crop_maping(v, p, size) for v, p in zip(view, param)])

    @staticmethod
    def crop_maping(view: torch.Tensor, p: torch.Tensor, size: int):
        p = tuple(p.cpu().numpy().astype(int))
        return transforms.functional.resized_crop(view, *p, size=(size, size))[None]

class Part_Whole_Loss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_weight: float = 100.0,
        pred_obj_path: Optional[str] = None,
        slot_prediction_path: Optional[str] = None,
        matched_idx_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "pred_obj_features": pred_obj_path,
                "slot_features": slot_prediction_path,
                "matched_idx": matched_idx_path,
            },
        )
        self.loss_weight = loss_weight
        self.temperature = 15
        self.unlabel_weight = 10

    @RoutableMixin.route
    def forward(
            self,
            pred_obj_features: torch.Tensor,  #[b,f,c,d]
            slot_features: torch.Tensor,
            matched_idx: dict,
    ):
        pred_obj_features = pred_obj_features.clone().detach()
        pred_obj_features = F.normalize(pred_obj_features, dim=-1)
        slot_features = F.normalize(slot_features, dim=-1)
        batch_size, num_frames, num_object, dim = pred_obj_features.shape
        num_slots = slot_features.shape[2]
        device = pred_obj_features.device
        loss = torch.zeros(1).to(device).mean()
        # merge [b, c]
        pred_obj_features = pred_obj_features.permute(1,0,2,3)
        slot_features = slot_features.permute(1,0,2,3)
        sums = 0
        for f in range(num_frames):
            if matched_idx[f]:
                for b in range(batch_size):
                    sim_scores = slot_features[f,b].mm(pred_obj_features[f,b].t())
                    pid_labels = torch.zeros((sim_scores.shape[0])).to(sim_scores.device)
                    pid_labels[pid_labels == 0] = -1
                    if str(b) in matched_idx[f]:
                        for i in range(num_object):
                            if str(i) in matched_idx[f][str(b)]:
                                for s_id in matched_idx[f][str(b)][str(i)]:
                                    pos_id = s_id
                                    pid_labels[pos_id] = i

                        p_i = F.softmax(sim_scores, dim=1)
                        # focal_p_i = F.log_softmax(sim_scores, dim=1)
                        focal_p_i = (1 - p_i + 1e-12) ** 2 * (p_i + 1e-12).log()
                        target = pid_labels.type(torch.LongTensor).to(focal_p_i.device)
                        loss_oim = F.nll_loss(focal_p_i, target,  ignore_index=-1)
                        loss_oim *= self.loss_weight
                        loss += loss_oim
                        sums += 1
        if sums == 0:
            return loss
        else:
            return loss / (1.0*(num_frames)*sums)


class AttnLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_weight: float = 100,
        attn_path: Optional[str] = None,
        tgt_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "segmentations": attn_path,
                "masks": tgt_path,
            },
        )
        self.loss_weight = loss_weight

    @RoutableMixin.route
    def forward(
            self,
            segmentations: torch.Tensor,
            masks: torch.Tensor,
    ):
        batch_size = segmentations.shape[0]
        total_seg_loss = 0.0
        for seg, mask in zip(segmentations, masks):

            seg = seg.permute(1, 0, 2, 3).flatten(1)
            mask = mask.permute(1, 0, 2, 3).flatten(1)
            mask_rate = mask.mean(1)
            index = torch.logical_and(mask_rate > 1e-4, mask_rate < 0.6)
            mask = mask[index]
            mask_num, temp_spatial_size = mask.shape
            seg = torch.clamp(seg, min=1e-7, max=1 - 1e-7)
            match_matrix = (
                    -(torch.log(seg) @ mask.T + torch.log(1 - seg) @ (1 - mask.T)) / temp_spatial_size
            )

            match = linear_sum_assignment(match_matrix.detach().cpu().numpy())
            row_ind, col_ind = match
            total_seg_loss += sum([match_matrix[i, j] for i, j in zip(row_ind, col_ind)]) / len(
                row_ind
            )
        total_seg_loss /= batch_size
        return total_seg_loss*self.loss_weight

class SupervisedRolloutLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_weight: float = 100,
        attn_path: Optional[str] = None,
        tgt_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "segmentations": attn_path,
                "masks": tgt_path,
            },
        )
        self.loss_weight = loss_weight

    @RoutableMixin.route
    def forward(
            self,
            segmentations: torch.Tensor,
            masks: torch.Tensor,
    ):
        batch_size = segmentations.shape[0]
        total_seg_loss = 0.0
        # only calucate frames [4:] loss
        segmentations = segmentations[:, 2:]
        masks = masks[:, 2:]
        masks = (masks > 0.5).float()
        for seg, mask in zip(segmentations, masks):

            seg = seg.permute(1, 0, 2, 3).flatten(1)
            mask = mask.permute(1, 0, 2, 3).flatten(1)
            mask_rate = mask.mean(1)
            index = torch.logical_and(mask_rate > 1e-4, mask_rate < 0.6)
            mask = mask[index]
            mask_num, temp_spatial_size = mask.shape
            seg = torch.clamp(seg, min=1e-7, max=1 - 1e-7)
            match_matrix = (
                    -(torch.log(seg) @ mask.T + torch.log(1 - seg) @ (1 - mask.T)) / temp_spatial_size
            )

            match = linear_sum_assignment(match_matrix.detach().cpu().numpy())
            row_ind, col_ind = match
            if len(row_ind) == 0:
                total_seg_loss += 0
            else:
                total_seg_loss += sum([match_matrix[i, j] for i, j in zip(row_ind, col_ind)]) / len(
                    row_ind
            )
        total_seg_loss /= batch_size
        return total_seg_loss*self.loss_weight

class SelfSupervisedRolloutLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_weight: float = 200,
        attn_path: Optional[str] = None,
        tgt_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "segmentations": attn_path,
                "masks": tgt_path,
            },
        )
        self.loss_weight = loss_weight

    @RoutableMixin.route
    def forward(
            self,
            segmentations: torch.Tensor,
            masks: torch.Tensor,
            smooth =1
    ):
        # only calucate frames [4:] loss
        # inputs = F.sigmoid(segmentations[:, 2:])

        b,f, c, h,w = segmentations.shape
        inputs = segmentations[:, 4:, 1:].reshape(-1, h, w)
        targets = masks[:, 4:, 1:].reshape(-1, h, w)
        targets = targets > 0.5
        ce_loss = F.binary_cross_entropy(inputs, targets.float())

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1).float()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1-dice
        total_loss = ce_loss + dice_loss


        return (total_loss) * self.loss_weight


class MemContrastiveLoss(nn.Module, RoutableMixin):
    def __init__(
        self,
        loss_weight: float = 1,
        merged_obj_path: Optional[str] = None,
        mem_path: Optional[str] = None,
        mem_table_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "pred_obj_features": merged_obj_path,
                "memory": mem_path,
                "mem_table": mem_table_path,   #[b,n,m]
            },
        )
        self.loss_weight = loss_weight
        self.temperature = 0.07

    @RoutableMixin.route
    def forward(
            self,
            pred_obj_features: torch.Tensor,  #[b,f,c,d]
            memory: torch.Tensor,
            mem_table: torch.Tensor,
    ):

        predictions = F.normalize(pred_obj_features, dim=-1)
        memory = F.normalize(memory, dim=-1)
        batch_size, num_frames, num_object, dim = predictions.shape
        device = predictions.device
        loss = torch.zeros(1).to(device).mean()

        for f in range(num_frames):
            for b in range(batch_size):
                for i in range(num_object):
                    pred = predictions[b, f, i].unsqueeze(0)
                    pos = mem_table[b,f,i].cpu().numpy().astype(int)
                    mem_pos = memory[b, f, 0, i].unsqueeze(0)
                    if i == 0:
                        mem_neg = memory[b,f,:pos, i+1:]
                    elif i == num_object:
                        mem_neg = memory[b,f,:pos, :i]
                    else:
                        mem_neg = torch.cat([memory[b,f,:pos,:i], memory[b,f,:pos,i+1:]], dim = -2)
                    mem_neg = mem_neg.reshape(-1, dim)
                    mem = torch.cat([mem_pos, mem_neg], dim=0)
                    mem = mem.reshape(-1, dim)
                    sim = pred.mm(mem.t())
                    sim /= self.temperature
                    pid_labels = torch.zeros(sim.shape[0], dtype=torch.long).to(device)
                    p_i = F.softmax(sim, dim=1)
                    # focal_p_i = F.log_softmax(sim_scores, dim=1)
                    focal_p_i = (1 - p_i + 1e-12) ** 2 * (p_i + 1e-12).log()
                    loss_oim = F.nll_loss(focal_p_i, pid_labels)
                    loss += loss_oim*self.loss_weight
        loss = loss / (1.0*(num_frames)*batch_size)
        return loss


class EM_seg_loss(nn.Module, RoutableMixin):
    def __init__(
            self,
            loss_weight: float = 20,
            attn_path: Optional[str] = None,
            tgt_path: Optional[str] = None,
            weights_path: Optional[str] = None,
            attn_index_path: Optional[str] = None,
            mem_table_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "segmentations": attn_path,
                "masks": tgt_path,
                "weights": weights_path,
                "attn_index": attn_index_path,
                "mem_table": mem_table_path,
            },
        )
        self.loss_weight = loss_weight

    @RoutableMixin.route
    def forward(
            self,
            segmentations: torch.Tensor,  # rollout_decode.masks
            masks: torch.Tensor,  # decoder.masks
            attn_index: torch.Tensor,
            mem_table: torch.Tensor,
            smooth=1
    ):
        # only calucate frames [4:] loss
        # inputs = F.sigmoid(segmentations[:, 2:])

        b, f, c, h, w = segmentations.shape
        # print("segmentations.shape:", segmentations.shape)
        # print("weights.shape:", weights.shape)
        device = segmentations.device

        _, _, n_slots, n_buffer = attn_index.shape

        # We will check the case for each
        all_ce_loss = 0
        counter = 0
        for n in range(n_buffer):
            for j in range(b):
                for k in range(1,f):
                    for s in range(n_slots):
                        index_prob = attn_index[j, k, s, n]
                        inputs = segmentations[j, k, n]
                        targets = masks[j, k, s] > 0.5
                        pos = mem_table[j, k, n]
                        targets = targets.float()
                        # if pos == 0:
                        #     targets = torch.ones(inputs.shape).to(inputs.device)

                        all_ce_loss += (index_prob * F.binary_cross_entropy(inputs, targets))

        total_loss = all_ce_loss
        return (total_loss) * self.loss_weight


class EM_rec_loss(nn.Module, RoutableMixin):
    def __init__(
            self,
            loss_weight: float = 20,
            attn_path: Optional[str] = None,
            rec_path: Optional[str] = None,
            tgt_path: Optional[str] = None,
            img_path: Optional[str] = None,
            tgt_vis_path: Optional[str] = None,
            weights_path: Optional[str] = None,
            attn_index_path: Optional[str] = None,
            slot_path: Optional[str] = None,
            pred_feat_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "segmentations": attn_path,
                "reconstructions": rec_path,
                "masks": tgt_path,
                "masks_vis": tgt_vis_path,
                "rec_tgt": img_path,
                "weights": weights_path,
                "attn_index": attn_index_path,
                "slots": slot_path,
                "pred_slots": pred_feat_path
            },
        )
        self.loss_weight = loss_weight
        self.loss_fn = (
            lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="sum") / x1.shape[0]
        )

    def rescale_mask(self, mask):
        max = torch.max(mask)
        min = torch.min(mask)
        mask_new = (mask-min) / (max - min)
        return mask_new

    @RoutableMixin.route
    def forward(
            self,
            segmentations: torch.Tensor,  # rollout_decode.masks
            masks: torch.Tensor,  # decoder.masks
            reconstructions: torch.Tensor,
            rec_tgt: torch.Tensor,
            masks_vis: torch.Tensor,
            attn_index: torch.Tensor,
            slots: torch.Tensor,
            pred_slots: torch.Tensor,
            smooth=1
    ):
        # only calucate frames [4:] loss
        # inputs = F.sigmoid(segmentations[:, 2:])

        b, f, c, h, w = segmentations.shape
        _, _, n_slots, n_buffer = attn_index.shape

        # We will check the case for each
        all_ce_loss = 0
        for n in range(n_buffer):
            for j in range(b):
                for k in range(1,f):
                    for s in range(n_slots):
                        index_prob = attn_index[j, k, s, n]
                        inputs = segmentations[j, k, n]
                        targets = masks[j, k, s] > 0.5
                        target_vis = masks_vis[j, k, s] > 0.5

                        rec_tgt_ = rec_tgt[j, k] * target_vis
                        rec_pred = reconstructions[j, k, n] * (target_vis)

                        # pos = mem_table[j, k, n]
                        targets = targets.float()

                        # loss = F.binary_cross_entropy(inputs, targets)+self.loss_fn(pred_slots[j,k,n], slots[j, k, s]) + 0.1*self.loss_fn(rec_pred, rec_tgt_)
                        # print(F.binary_cross_entropy(inputs, targets))
                        loss= F.binary_cross_entropy(inputs, targets) + 0.1*self.loss_fn(rec_pred, rec_tgt_)


                        all_ce_loss += (index_prob * loss)

        total_loss = all_ce_loss / (b*(f-1)*n_buffer*n_slots)
        return (total_loss) * self.loss_weight

class EM_rec_loss_simplify(nn.Module, RoutableMixin):
    def __init__(
            self,
            loss_weight: float = 20,
            attn_path: Optional[str] = None,
            rec_path: Optional[str] = None,
            tgt_path: Optional[str] = None,
            img_path: Optional[str] = None,
            tgt_vis_path: Optional[str] = None,
            weights_path: Optional[str] = None,
            attn_index_path: Optional[str] = None,
            slot_path: Optional[str] = None,
            pred_feat_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "segmentations": attn_path,
                "reconstructions": rec_path,
                "masks": tgt_path,
                "masks_vis": tgt_vis_path,
                "rec_tgt": img_path,
                "weights": weights_path,
                "attn_index": attn_index_path,
                "slots": slot_path,
                "pred_slots": pred_feat_path
            },
        )
        self.loss_weight = loss_weight
        self.loss_fn = (
            lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="none")
        )

    @RoutableMixin.route
    def forward(
            self,
            segmentations: torch.Tensor,  # rollout_decode.masks
            masks: torch.Tensor,  # decoder.masks
            reconstructions: torch.Tensor,
            rec_tgt: torch.Tensor,
            masks_vis: torch.Tensor,
            attn_index: torch.Tensor,
            slots: torch.Tensor,
            pred_slots: torch.Tensor,
            smooth=1
    ):
        b, f, c, h, w = segmentations.shape
        _, _, n_slots, n_buffer = attn_index.shape

        segmentations = segmentations.reshape(-1, n_buffer, h, w).unsqueeze(1).repeat(1,n_slots,1,1,1)
        masks = masks.reshape(-1, n_slots, h, w).unsqueeze(2).repeat(1,1,n_buffer,1,1)
        masks = masks > 0.5
        masks_vis = masks_vis.reshape(-1, n_slots, h, w).unsqueeze(2).unsqueeze(3).repeat(1,1,n_buffer,3,1,1)
        masks_vis = masks_vis > 0.5
        attn_index = attn_index.reshape(-1, n_slots, n_buffer)
        rec_tgt = rec_tgt.reshape(-1,3,h,w).unsqueeze(1).unsqueeze(2).repeat(1,n_slots,n_buffer,1,1,1)
        reconstructions = reconstructions.reshape(-1, n_buffer, 3, h, w).unsqueeze(1).repeat(1,n_slots,1,1,1,1)
        rec_pred = reconstructions * masks_vis
        rec_tgt_ = rec_tgt * masks_vis
        loss = torch.sum(F.binary_cross_entropy(segmentations, masks.float(), reduction = 'none'), (-1,-2)) / (h*w) + 0.1 * torch.sum(self.loss_fn(rec_pred, rec_tgt_), (-3,-2,-1))
        # be = torch.sum(F.binary_cross_entropy(segmentations, masks.float(), reduction = 'none'), (-1,-2)) / (h*w)
        # rec = 0.1 * torch.sum(self.loss_fn(rec_pred, rec_tgt_), (-3,-2,-1))
        # print(be[0,1,1], rec[0,1,1])
        total_loss = torch.sum(attn_index * loss, (0,1,2)) / (b * f * n_slots * n_buffer)
        return (total_loss) * self.loss_weight

class SegLoss(nn.Module, RoutableMixin):
    def __init__(
            self,
            loss_weight: float = 20,
            attn_path: Optional[str] = None,
            tgt_path: Optional[str] = None,
            weights_path: Optional[str] = None,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "segmentations": attn_path,
                "masks": tgt_path,
                "weights": weights_path,
            },
        )
        self.loss_weight = loss_weight

    @RoutableMixin.route
    def forward(
            self,
            segmentations: torch.Tensor,  # decoder.masks
            masks: torch.Tensor,  # mem_masks.masks
            weights: torch.Tensor,
            smooth=1
    ):

        b, f, c, h, w = segmentations.shape
        device = segmentations.device
        masks_new = torch.zeros(masks.shape).to(device)
        for j in range(b):
            for k in range(f):
                index = torch.nonzero(weights[j][k])
                for i in range(len(index)):
                    masks_new[j][k][i] = masks[j][k][index[i][1]]

        inputs = segmentations[:, 1:, 0:].reshape(-1, h, w)
        targets = masks_new[:, 1:, 0:].reshape(-1, h, w)
        targets = targets > 0.5
        ce_loss = F.binary_cross_entropy(inputs, targets.float())

        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1).float()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice
        total_loss = ce_loss + dice_loss
        return (total_loss) * self.loss_weight





