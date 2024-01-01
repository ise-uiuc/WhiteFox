
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.module1 = torch.nn.Conv2d(196, 48, kernel_size=(1, 1), stride=(1, 1))
        self.module2 = torch.nn.ReLU()
        self.module3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.module1(x1)
        v2 = self.module2(v1)
        v3 = self.module3(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = 0.002
max = 0.003
# Inputs to the model
x1 = torch.randn(1, 196, 87, 73)
