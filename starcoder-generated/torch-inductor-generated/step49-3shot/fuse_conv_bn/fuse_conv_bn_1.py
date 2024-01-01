
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.conv1 = torch.nn.Conv1d(1, 1, 3, stride=2)
        torch.manual_seed(3)
        self.conv2 = torch.nn.Conv1d(1, 1, 3, stride=2)
        self.bn = torch.nn.BatchNorm1d(1)
        self.relu = torch.nn.ReLU()
        self.mp = torch.nn.MaxPool1d(2)
    def forward(self, input):
        c = self.conv1(input)
        a = self.bn(c)
        d = self.relu(c)
        e = self.conv2(d)
        b = self.mp(e)
        return b
# Inputs to the model
input = torch.randn(1, 1, 7)
