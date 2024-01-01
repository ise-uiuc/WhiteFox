
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v3 = self.conv2(v1)
        v4 = F.relu(v3 - v1)
        v5 = F.relu(v3)
        return torchvision.ops.roi_align(v5, None, 1, 0, 1, 1, True)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
