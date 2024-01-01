
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Conv2d(32, 64, 3, stride=2)
        self.relu1 = torch.nn.ReLU()
        self.m2 = torch.nn.Conv2d(64, 1, 3, stride=1)

    def forward(self, x):
        return self.m2(self.relu1(self.m1(x)))
# Inputs to the model
x = torch.randn(8, 32, 32, 32)
