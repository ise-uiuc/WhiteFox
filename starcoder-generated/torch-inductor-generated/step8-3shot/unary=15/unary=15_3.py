
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.mean()
        v2 = x1.min()
        v3 = x1.max()
        return v1, v2, v3
# Inputs to the model
x1 = torch.randn(4, 4)
