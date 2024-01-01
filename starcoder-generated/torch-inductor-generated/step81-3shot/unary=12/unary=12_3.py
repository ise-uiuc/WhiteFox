
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.avgpool(x1)
        v2 = v1 * torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
