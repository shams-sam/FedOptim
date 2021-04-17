import torch
import torch.nn as nn


# https://github.com/HaotianMXu/Multiclass_LinearSVM_with_SGD/
# blob/master/linearSVM.py
class multiClassHingeLoss(nn.Module):
    def __init__(self, p=2, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p = p
        self.margin = margin
        self.weight = weight
        self.size_average = size_average

    def forward(self, output, y):
        output_y = output[
            torch.arange(0, y.size()[0]).long().cuda(),
            y.data.cuda()].view(-1, 1)
        loss = output - output_y + self.margin
        loss[
            torch.arange(0, y.size()[0]).long().cuda(),
            y.data.cuda()] = 0
        loss = nn.functional.relu(loss)
        loss = torch.pow(loss, self.p)

        if(self.weight is not None):
            loss = loss*self.weight

        loss=torch.sum(loss)
        if(self.size_average):
            loss /= output.size()[0]
        return loss
