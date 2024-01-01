
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(96, 96, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(96, 12, 1, stride=1, padding=0) # Note that we use a 1x1 pointwise convolution filter here
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 - 0.5
        v5 = F.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 96, 64, 64)
