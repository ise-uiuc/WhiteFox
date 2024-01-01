
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, target, weight):
        x = x * target.max(1)[0].reshape(-1, 1, 1) * weight
        return x
# Inputs to the model
x = torch.randn(1)
target = torch.randn(1)
weight = torch.randn(1)
