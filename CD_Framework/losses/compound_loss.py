from losses.bce import *
from losses.dice import *


class BceDiceLoss(nn.Module):
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        self.bce = BceLoss() # bce计算损失
        self.dice = DiceLoss() # dice计算损失

    def forward(self, logits, true):

        return self.bce(logits, true) + self.dice(logits, true)
