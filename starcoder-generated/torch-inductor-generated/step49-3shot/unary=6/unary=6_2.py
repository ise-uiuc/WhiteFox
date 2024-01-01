
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten(1, 3)
        self.linear = torch.nn.Linear(5408, 256)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=1)
        self.fc = torch.nn.Linear(256, 1)
    def forward(self, x1):
        t1 = self.linear(self.flatten(self.conv(x1)))
        t2 = self.avgpool(t1)
        t3 = self.fc(torch.flatten(t2, 1))
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
