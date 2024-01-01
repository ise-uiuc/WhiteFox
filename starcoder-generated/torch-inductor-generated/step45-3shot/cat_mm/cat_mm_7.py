
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.cat([x1, x1], 1)
        for loopVar1 in range(5):
            t2 = torch.cat([t1, t1], 1)
        return torch.cat([t2, t2, t2, t2], 1)
# Inputs to the model
x1 = torch.randn(2, 4)
