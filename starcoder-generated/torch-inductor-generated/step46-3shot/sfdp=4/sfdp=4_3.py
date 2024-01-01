
class Module1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x + x
        x = x + x
        return x
class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 1)
        self.conv2 = Module1()
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
model = Module0()
x = torch.randn(1, 4, 12, 12)
y = model(x)
# The input tensor to model ends

# Model begins
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(18, 36)
        self.linear2 = torch.nn.Linear(36, 72)
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x
# Input tensor to model
x = torch.randn(1, 18)
y = model(x)
