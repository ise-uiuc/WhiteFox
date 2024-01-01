
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bias = torch.nn.Parameter(torch.randn(1))
 
    def forward(self, x):
        v1 = torch.tanh(self.conv(x))
        v2 = torch.sigmoid(self.conv2(x))
        v3 = torch.matmul(v1, v2.) + self.bias
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
