
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 7, stride=2, padding=2, dilation=3)
        self.conv2 = torch.nn.Conv2d(1, 1, 7, stride=1, padding='same', dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model - Tensor of size (16, 1, 224, 224)
x1 = torch.randn(16, 1, 224, 224)
