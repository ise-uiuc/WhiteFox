
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.conv1 = torch.nn.Conv1d(3, 3, 3)
        self.block = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.ReLU())
    def forward(self, x2):
        y1 = self.conv1(x2)
        y2 = self.block(y1)
        return y2
# Inputs to the model
x2 = torch.randn(1, 3, 10)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Conv2d(7, 3, 1))
    def forward(self, x):
        return self.block(x)
# Inputs to the model
x = torch.randn(1, 7, 10, 10)
