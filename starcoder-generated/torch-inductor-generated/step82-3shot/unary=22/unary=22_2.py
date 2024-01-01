
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3, x4):
        v1 = self.conv(torch.cat((x1, x2, x3, x4), 1))
        v2 = self.linear(v1)
        v3 = torch.tanh(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(5, 128, 14, 14)
x3 = torch.randn(8, 512, 7, 7)
x4 = torch.randn(16, 2048, 3, 3)
y = torch.randn(1, 1)
