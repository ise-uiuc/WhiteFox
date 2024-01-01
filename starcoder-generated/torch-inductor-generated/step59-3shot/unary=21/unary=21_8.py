
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1D = torch.nn.Conv1d(144, 300, 3, padding=1)
        self.conv2D = torch.nn.Conv2d(300, 288, (1, 6), padding=0, stride=1)
        self.batchNorm2D = torch.nn.BatchNorm2d(288)
        self.linear = torch.nn.Linear(288, 1)
    def forward(self, x):
        x0 = self.conv1D(x)
        x1 = torch.tanh(x0)
        x2 = torch.mean(x1, dim=2)
        x3 = x2.view(-1, 1, 300, 11)
        x4 = self.conv2D(x3)
        x5 = self.batchNorm2D(x4)
        x6 = torch.tanh(x5)
        x7 = torch.mean(x6, dim=2)
        x8 = self.linear(x7)
        x9 = torch.tanh(x8)
        return x9
# Inputs to the model
x = torch.randn(1, 144, 20, 10)
