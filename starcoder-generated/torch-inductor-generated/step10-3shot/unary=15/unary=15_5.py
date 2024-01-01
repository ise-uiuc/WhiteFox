
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = torch.nn.AdaptiveAvgPool2d(output_size=1)
    def forward(self, x1):
        v1 = self.avg(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
