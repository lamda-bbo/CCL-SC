
import torch
from torch.nn import functional as F


def deep_gambler_loss(outputs, targets, reward):
    outputs = F.softmax(outputs, dim=1)
    outputs, reservation = outputs[:,:-1], outputs[:,-1]
    # gain = torch.gather(outputs, dim=1, index=targets.unsqueeze(1)).squeeze()
    gain = outputs[torch.arange(targets.shape[0]), targets]
    doubling_rate = (gain.add(reservation.div(reward))).log()
    return -doubling_rate.mean()

def log_margin_loss(outputs, targets, margin_alpha):
    outputs = F.softmax(outputs, dim=1)
    # outputs = F.log_softmax(outputs, dim=1)
    max_runup = torch.topk(outputs,k=2,dim=1)
    max_runup = max_runup[1]
    top1_mask = max_runup[:,0].view(-1,1)
    targets = targets.view(-1,1)
    correct_mask = targets==top1_mask
    correct_mask = correct_mask.long()
    error_mask = 1.0 - correct_mask 
    
    top2_mask = max_runup[:,1].view(-1,1)
    top1 = outputs.gather(1,top1_mask)
    top2 = outputs.gather(1,top2_mask)
    correct = outputs.gather(1,targets)
    # print(top2.sum(),top2.shape)
    loss_correct = -(torch.log(top2+1)*correct_mask).sum()/(correct_mask.sum() + 1)
    # loss_error = ((torch.log(correct+1) - torch.log(top1+1) )*error_mask).sum()
    loss_error = ((- torch.log(top1+1) )*error_mask).sum()/(error_mask.sum() + 1)
    return - margin_alpha*loss_correct - margin_alpha*loss_error

class SelfAdativeTraining():
    def __init__(self, num_examples=50000, num_classes=10, mom=0.9):
        self.prob_history = torch.zeros(num_examples, num_classes)
        self.updated = torch.zeros(num_examples, dtype=torch.int)
        self.mom = mom
        self.num_classes = num_classes

    def _update_prob(self, prob, index, y):
        onehot = torch.zeros_like(prob)
        onehot[torch.arange(y.shape[0]), y] = 1
        prob_history = self.prob_history[index].clone().to(prob.device)

        # if not inited, use onehot label to initialize runnning vector
        cond = (self.updated[index] == 1).to(prob.device).unsqueeze(-1).expand_as(prob)
        prob_mom = torch.where(cond, prob_history, onehot)

        # momentum update
        prob_mom = self.mom * prob_mom + (1 - self.mom) * prob

        self.updated[index] = 1
        self.prob_history[index] = prob_mom.to(self.prob_history.device)

        return prob_mom

    def __call__(self, logits, y, index):
        prob = F.softmax(logits.detach()[:, :self.num_classes], dim=1)
        prob = self._update_prob(prob, index, y)

        soft_label = torch.zeros_like(logits)
        soft_label[torch.arange(y.shape[0]), y] = prob[torch.arange(y.shape[0]), y]
        soft_label[:, -1] = 1 - prob[torch.arange(y.shape[0]), y]
        soft_label = F.normalize(soft_label, dim=1, p=1)
        loss = torch.sum(-F.log_softmax(logits, dim=1) * soft_label, dim=1)
        return torch.mean(loss)
