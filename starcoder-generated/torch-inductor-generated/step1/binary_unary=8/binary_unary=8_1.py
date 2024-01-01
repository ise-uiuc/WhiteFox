
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bias = torch.nn.Parameter(torch.randn(1, 8, 64, 64))
 
    def forward(self, x):
        v1 = self.conv(x) + self.bias
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
