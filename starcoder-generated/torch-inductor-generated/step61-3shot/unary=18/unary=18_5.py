
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = F.avg_pool2d(x1, 14, stride=2, padding=3)
        return v1
# Inputs to the model
x1 = torch.randn(1, 11, 64, 64)
