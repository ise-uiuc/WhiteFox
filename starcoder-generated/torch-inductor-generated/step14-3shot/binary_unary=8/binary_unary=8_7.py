
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(3, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
# Input shape [1, 3, 880] - (batch, channel, time)
x1 = torch.randn(1, 3, 880)
