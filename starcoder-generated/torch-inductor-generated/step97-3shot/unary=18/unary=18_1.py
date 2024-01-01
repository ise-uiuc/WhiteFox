
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.avg_pool2d(x1, 4, 2, 0)
        v2 = torch.nn.functional.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 100, 100, 100)
