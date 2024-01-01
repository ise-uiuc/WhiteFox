
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 3)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(2, 2, 1)
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
