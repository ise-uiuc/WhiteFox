
class Model(torch.nn.Module):
    def __init__(self, loopVar):
        super().__init__()
        self.loopVar = loopVar
    def forward(self, x1, x2):
        v = []
        return torch.cat(v, 1)
loopVar = 0
# Inputs to the model
x1 = torch.randn(3, 2)
x2 = torch.randn(3, 2)
