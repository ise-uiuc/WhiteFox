
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1) # Output channel number is 8
        self.conv2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1) # Output channel number is also 8
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1) # Output channel number is also 8
 
 
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        x4 = v1 + x2
        x5 = x3 + v3
        v4 = v1 * x4
        v5 = v2 * x5
        v6 = v3 * v4
        v7 = v3 * v5
        return v6, v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
x3 = torch.randn(1, 8, 64, 64)
