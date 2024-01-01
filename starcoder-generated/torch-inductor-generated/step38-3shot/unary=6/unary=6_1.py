
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.size()
        return int(v1[-1])
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
