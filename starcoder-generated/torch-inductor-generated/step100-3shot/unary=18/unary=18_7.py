
class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6):
        return torch.nn.MaxPool1d(x4, **{'kernel_size': 2,'stride': x1}), torch.nn.MaxPool2d(**{'kernel_size': x3,'stride': x2}), torch.nn.MaxPool3d(**{'kernel_size': (x4, x5, 5),'stride': x6})
# Inputs to the model
x1 = 3
x2 = 3
x3 = 4
x4 = 3
x5 = 1
x6 = 3
