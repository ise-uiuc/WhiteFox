
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv1d(2, 2, 2)
        self.bn1 = torch.nn.BatchNorm1d(2)
        self.conv2 = torch.nn.Conv1d(2, 2, 2)
        self.bn2 = torch.nn.BatchNorm1d(2)
        self.conv3 = torch.nn.Conv1d(2, 2, 3)
        self.bn3 = torch.nn.BatchNorm1d(2)
    def forward(self, x1):
        x1 = self.relu(self.bn1(self.conv1(x1)))
        x1 = self.relu(self.bn2(self.conv2(x1)))
        x1 = self.relu(self.bn3(self.conv3(x1)))
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 3)
