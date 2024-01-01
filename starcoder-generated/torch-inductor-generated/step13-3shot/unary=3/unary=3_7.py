
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 7, stride=7, padding=3)
    def forward(self, x1):
        v1 = torch.randn(1, 1, 224, 224)
        return self.conv1(x1)
# Inputs to the model
x1 = torch.randn(1, 1, 416, 416)
