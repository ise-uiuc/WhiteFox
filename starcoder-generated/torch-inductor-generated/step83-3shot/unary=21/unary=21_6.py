
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(4, 4, 2, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(4, 2, 3, stride=1, padding=1)
        self.tanh1 = torch.nn.Tanh()
        self.tanh2 = torch.nn.Tanh()
        self.tanh3 = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.relu1(x)
        x2 = self.conv(x1)
        x3 = self.relu2(x2)
        x4 = self.conv1(x3)
        x5 = self.tanh1(x4)
        x6 = self.tanh1(x5)
        x7 = self.tanh1(x6)
        x8 = self.tanh1(x7)
        x9 = self.tanh2(x8)
        return x9
# Inputs to the model
x = torch.randn(1, 4, 8, 8)
