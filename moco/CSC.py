
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):

    def __init__(self, dim=128, K=65536, m=0.999, T=0.07, num_class = 10, mlp=False):
        super(MoCo, self).__init__()
        self.base_temperature = 0.10
        self.K = K
        self.K2 = K
        self.m = m
        self.T = T

        # create the error queue
        self.register_buffer("queue", torch.randn(dim, self.K2))
        self.queue = nn.functional.normalize(self.queue, dim=0)
         # create the prediction queue
        self.register_buffer("prediction_queue", torch.randint(low=0, high=num_class, size=(self.K2,), dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the correct queue
        self.register_buffer("correct_queue", torch.randn(dim, K))
        self.correct_queue = nn.functional.normalize(self.correct_queue, dim=0)
        
        # create the correct prediction queue
        self.register_buffer("correct_prediction_queue", torch.randint(low=0, high=num_class, size=(K,), dtype=torch.long))
        
        # create the correct queue pointer
        self.register_buffer("correct_queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, k_error, k_correct, correct_predicts, error_predicts):
        batch_size = k_error.shape[0]

        ptr = int(self.queue_ptr)
        full_k1 = False
        full_k2 = False

        if ptr + batch_size >= self.K2:
            self.queue[:, ptr : self.K2] = k_error[:self.K2 - ptr, :].T
            self.queue[:,  : batch_size - self.K2 + ptr] = k_error[self.K2 - ptr :, :].T
            self.prediction_queue[ptr : self.K2] = error_predicts[:self.K2 - ptr]
            self.prediction_queue[: batch_size - self.K2 + ptr] = error_predicts[self.K2 - ptr :]
            ptr = batch_size - self.K2 + ptr  # move pointer
            full_k1 = True
        else:
            self.queue[:, ptr : ptr + batch_size] = k_error.T
            
            self.prediction_queue[ptr : ptr + batch_size] = error_predicts
            ptr = (ptr + batch_size) % self.K2  # move pointer

        self.queue_ptr[0] = ptr

        batch_size = correct_predicts.shape[0]

        ptr = int(self.correct_queue_ptr)
        if ptr + batch_size >= self.K:
            self.correct_queue[:, ptr : self.K] = k_correct[:self.K - ptr, :].T
            self.correct_queue[:,  : batch_size - self.K + ptr] = k_correct[self.K - ptr :, :].T
            self.correct_prediction_queue[ptr : self.K] = correct_predicts[:self.K - ptr]
            self.correct_prediction_queue[: batch_size - self.K + ptr] = correct_predicts[self.K - ptr :]
            ptr = batch_size - self.K + ptr  # move pointer
            full_k2 = True
        else:
            self.correct_queue[:, ptr : ptr + batch_size] = k_correct.T
            self.correct_prediction_queue[ptr : ptr + batch_size] = correct_predicts
            ptr = (ptr + batch_size) % self.K  # move pointer

        self.correct_queue_ptr[0] = ptr
        return full_k1, full_k2

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, hidden_features, hidden_features_k, targets, outputs, outputs_k, epoch, pretrain, full_flag):

        predicted_k = outputs_k.argmax(dim=1)
        correct_mask_k = (predicted_k == targets.cuda())
        correct_predicts_k = predicted_k[correct_mask_k]
        error_predicts_k = predicted_k[~correct_mask_k]
        predicted = outputs.argmax(dim=1)
        outputs = F.softmax(outputs, dim=1)
        sr = outputs.max(1).values.detach()
        correct_mask = (predicted == targets.cuda())
        correct_predicts = predicted[correct_mask]
        error_predicts = predicted[~correct_mask]
        
        full_k1 = False
        full_k2 = False
        # compute error key features
        with torch.no_grad():  # no gradient to keys

            k_error = hidden_features_k[~correct_mask_k].cuda()  # keys: NxC
            k_error = nn.functional.normalize(k_error, dim=1)

            k_correct = hidden_features_k[correct_mask_k].cuda() 
            k_correct = nn.functional.normalize(k_correct, dim=1)

        if epoch >= pretrain and full_flag:
            # compute query features
            q = hidden_features.cuda()  # queries: NxC
            q = nn.functional.normalize(q, dim=1)

            #TF, not sim
            sim_matrix = torch.matmul(q, self.queue.detach().cuda())
            eq_matrix = torch.eq(targets.view(-1, 1), self.prediction_queue.detach().view(1, -1).cuda())
            sim_matrix *= eq_matrix
            sim_matrix += (eq_matrix == False).float() * -1e9
            del eq_matrix 

            #TT, sim
            sim_matrix_tp = torch.matmul(q, self.correct_queue.detach().cuda())
            eq_matrix_tp = torch.eq(targets.view(-1, 1), self.correct_prediction_queue.detach().view(1, -1).cuda())
            sim_matrix_tp *= eq_matrix_tp
            sim_matrix_tp = sim_matrix_tp[eq_matrix_tp]
            non_zero_counts = (eq_matrix_tp != 0).sum(dim=1).cuda()
            expanded_non_zero_counts = (non_zero_counts / sr).repeat_interleave(non_zero_counts)
            del eq_matrix_tp

            expanded_sim_matrix = sim_matrix.repeat_interleave(non_zero_counts, dim=0)
            del sim_matrix
            logits_t = torch.cat([sim_matrix_tp.unsqueeze(-1), expanded_sim_matrix], dim=1)
            logits_t /= self.T
            logits_t = logits_t - logits_t.max(dim=1, keepdim=True).values  # 防止溢出
            exp_logits_t = logits_t.exp()
            logsumexp_t = exp_logits_t.sum(dim=1).log()
            del logits_t, exp_logits_t
            # InfoNCE loss
            info_nce_loss_t = logsumexp_t - sim_matrix_tp
            info_nce_loss_t /= expanded_non_zero_counts
            info_nce_loss_t = torch.sum(info_nce_loss_t) / q.shape[0]

            # part 2 not used
            if error_predicts.shape[0]==0 or True:
                # print('0error')
                return full_k1, full_k2, sum(correct_mask_k).item() / int(targets.size().numel()), (self.T / self.base_temperature) * info_nce_loss_t

            fq = hidden_features[~correct_mask].cuda()  
            fq = nn.functional.normalize(fq, dim=1)

            sim_matrix_ft = torch.matmul(fq, self.correct_queue.detach().cuda())
            eq_matrix_ft = torch.eq(error_predicts.view(-1, 1), self.correct_prediction_queue.detach().view(1, -1).cuda())
            sim_matrix_ft *= eq_matrix_ft
            sim_matrix_ft += (eq_matrix_ft == False).float() * -1e9

            # FF, sim
            sim_matrix_ff = torch.matmul(fq, self.correct_queue.detach().cuda())
            eq_matrix_ff = torch.eq(targets[~correct_mask].view(-1, 1), self.correct_prediction_queue.detach().view(1, -1).cuda())
            sim_matrix_ff *= eq_matrix_ff

            # sim_matrix_ff = ((sim_matrix_ff).sum(1) + 1e-5)/ (eq_matrix_ff.sum(1) + 2e-8)       


            sim_matrix_ff = sim_matrix_ff[eq_matrix_ff]
            non_zero_counts = (eq_matrix_ff != 0).sum(dim=1)
            expanded_non_zero_counts = non_zero_counts.repeat_interleave(non_zero_counts)
            expanded_sim_matrix_ft = sim_matrix_ft.repeat_interleave(non_zero_counts, dim=0)
            logits_f = torch.cat([sim_matrix_ff.unsqueeze(-1), expanded_sim_matrix_ft], dim=1)
            logits_f /= self.T
            logits_f = logits_f - logits_f.max(dim=1, keepdim=True).values  # 防止溢出
            exp_logits_f = logits_f.exp()
            logsumexp_f = exp_logits_f.sum(dim=1).log()

            # InfoNCE loss
            info_nce_loss_f = logsumexp_f - sim_matrix_ff
            info_nce_loss_f /= expanded_non_zero_counts
            info_nce_loss_f = torch.sum(info_nce_loss_f) / error_predicts.shape[0]
            # logits_t = torch.cat([sim_matrix_tp.unsqueeze(-1), sim_matrix], dim=1)
            # logits_f = torch.cat([sim_matrix_ff.unsqueeze(-1), sim_matrix_ft], dim=1)
            # logits = torch.cat([logits_t, logits_f], dim=0)
            # logits /= self.T
            # # labels: positive key indicators

            full = self._dequeue_and_enqueue(k_error, k_correct, correct_predicts_k, error_predicts_k)
            return full, sum(correct_mask_k).item() / int(targets.size().numel()), (self.T / self.base_temperature) *(info_nce_loss_t + info_nce_loss_f)
            # dequeue and enqueue
        
        full_k1, full_k2 = self._dequeue_and_enqueue(k_error, k_correct, correct_predicts_k, error_predicts_k)
        return full_k1, full_k2, sum(correct_mask_k).item() / int(targets.size().numel())

        


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output