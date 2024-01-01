
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 0.5
        v4 = torch.transpose(v3, 0, 1)
        v5 = F.relu(v4)
        v6 = v5[:, 0, :, :]
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
