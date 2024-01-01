
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(7, 7, 3)
        self.conv2 = torch.nn.Conv1d(7, 8, 3)
        torch.manual_seed(1)
        self.bn1 = torch.nn.BatchNorm1d(7)
        torch.manual_seed(0)
        self.bn2 = torch.nn.BatchNorm1d(8)
        torch.manual_seed(1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        t1 = self.relu(self.bn1(self.conv1(x1)))
        t2 = self.relu(self.bn2(self.conv2(t1)))
        y = torch.tanh(t2)
        return (t1, t2, y)
# Inputs to the model
x1 = torch.randn(1, 7, 6)
