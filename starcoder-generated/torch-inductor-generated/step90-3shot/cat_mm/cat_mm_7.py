
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        for loopVar1 in range(1):
            x1 = torch.mm(x2, x1)
        return torch.mm(x1, x2)
# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 10)
