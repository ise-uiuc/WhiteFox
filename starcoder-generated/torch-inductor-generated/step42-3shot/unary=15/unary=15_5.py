
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = torch.transpose(v2, 1, 0)
        v4 = torch.transpose(v2, 1, 3)
        v32 = self.conv2(v3)
        v42 = self.conv2(v4)
        v33 = torch.transpose(v32, 0, 2)
        v43 = torch.transpose(v42, 0, 2)
        v5 = torch.cat([v33, v43], 0)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
