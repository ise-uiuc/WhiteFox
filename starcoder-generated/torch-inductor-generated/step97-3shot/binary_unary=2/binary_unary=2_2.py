
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv1d(8, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = torch.transpose(v3, 1, 2)
        v5 = self.conv2(v4)
        v6 = v5 - 0.5
        v7 = F.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
