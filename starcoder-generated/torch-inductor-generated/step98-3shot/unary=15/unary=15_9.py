
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 2, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv1d(2, 2, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv1d(2, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        return torch.tanh(v3)
# Inputs to the model
x1 = torch.randn(1, 3, 33)
