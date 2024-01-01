
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        newList = [] # newList is introduced as v1, v2, v3 would be deleted by torch.cat 
        return torch.cat([newList[5, 4], newList[2], newList[1]], 1)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 2)
