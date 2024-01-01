
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=1)
 
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 32, 32)
