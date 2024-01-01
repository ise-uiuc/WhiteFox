
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = F.avg_pool2d(x1, 4, 4)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
