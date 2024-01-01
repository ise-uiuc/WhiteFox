
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.relu(v1)
        v1_1 = F.interpolate(v1, scale_factor=1.0, recompute_scale_factor=None, mode='nearest')
        v2 = self.conv2(v1_1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 128, 128)
