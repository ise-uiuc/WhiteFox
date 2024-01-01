
import datetime
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        tensorList = []
        for loopVar1 in range(10):
            v = x1.type_as(torch.DoubleTensor()).to(torch.device("cuda:" + str((loopVar1 % 4))))
            v = x1 + (v + torch.randn(torch.Size([1, 3])))
            v = torch.mm(v, v)
            v = torch.mm(v, v)
            v = torch.mm(v, v)
            v = torch.mm(v, v)
            tensorList.append(v.type_as(torch.FloatTensor()).to(torch.device("cuda:0" if (loopVar1 %2) == 0 else "cuda:1")))
        return torch.cat(tensorList, 0)
# Inputs to the model
x1 = torch.randn(2, 2, requires_grad=True)
