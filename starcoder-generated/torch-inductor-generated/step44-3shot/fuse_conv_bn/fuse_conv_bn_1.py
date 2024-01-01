
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 3, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(3, 3, 12)
        self.relu2 = torch.nn.ReLU()
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.relu1(s)
        y = self.conv2(t)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 74)
