
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.max_pool2d(v1, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
