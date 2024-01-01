
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 8, 3, stride=1)
        self.conv2 = torch.nn.Conv1d(8, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv1d(16, 32, 3, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x2 = torch.randn(1, 3, 1728)
