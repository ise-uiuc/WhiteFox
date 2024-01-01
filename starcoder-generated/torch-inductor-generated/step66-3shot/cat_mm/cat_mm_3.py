
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        for loopVar1 in range(5):
            v = v + v
            c = v * v 
        return torch.cat([torch.mm(x1, x2), torch.mm(x1, x2), c], 1) 
# Inputs to the model
x1 = torch.randn(5, 2)
x2 = torch.randn(2, 2)
