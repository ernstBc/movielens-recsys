import torch
from torch import nn



class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        # Create a binary mask where target is not zero (observed data)
        mask = (target != 0).float()

        # Calculate the squared error 
        squared_errors = (pred - target) ** 2

        # Apply a mask to ignore zero values 
        masked_errors = squared_errors * mask
        loss_sum = masked_errors.sum()

        # Average of non zeroes values (+ epsilon to avoid division by zero)
        total_observed = mask.sum()
        loss = loss_sum / (total_observed + 1e-8)

        return loss


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super().__init__()
        self.gamma = gamma

    def forward(self, pos_scores, neg_scores):
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores + self.gamma)).mean()
        return loss


def negative_sampling_collate_fn(batch):
    user_ids, items_ids, ratings, neg_items_ids = zip(*batch)
    neg_len = len(neg_items_ids[0])

    items_ = []
    rating_ = []
    user_ = []
    for user, item, neg_items, rating in zip(user_ids, items_ids, neg_items_ids, ratings):
        i = [item.item()] + neg_items.tolist()
        r = [rating.item()] + [0] * neg_len
        u = [user.item()] * (neg_len + 1)
        
        rating_.append(r)
        items_.append(i)
        user_.append(u)

    user_tensor = torch.tensor(user_, dtype=torch.long)
    item_tensor = torch.tensor(items_, dtype=torch.long)
    rating_tensor = torch.tensor(rating_, dtype=torch.float)

    return user_tensor, item_tensor, rating_tensor