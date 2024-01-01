
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 5, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
    def forward(self, inputs1):
        v1 = self.conv1(inputs1)
        v2 = v1 - torch.mean(v1)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - torch.min(v4)
        v6 = F.relu(v5)
        return v6
# Inputs to the model
inputs1 = torch.randn(1, 3, 256, 256)
