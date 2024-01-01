
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(6, 3, 1)
        self.relu = torch.nn.ReLU()
        self.b = torch.nn.BatchNorm2d(6)
        self.maxpool2d = torch.nn.MaxPool2d((1, 1))
    def forward(self, x4):
        x4 = self.b(x4)
        x4 = self.maxpool2d(x4)
        return self.relu(self.a(x4))
# Inputs to the model
x4 = torch.randn(10, 6, 38, 32)
