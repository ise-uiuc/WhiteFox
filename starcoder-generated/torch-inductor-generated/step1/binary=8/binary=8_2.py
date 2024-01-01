
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
        self.other = torch.Tensor([1])
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.other + v1
        return v2

# Initializing the model
m = Model()

# Initializing the weights
m.conv.weight = torch.nn.Parameter(torch.ones_like(m.conv.weight) / 0.01)
m.other.data = torch.Tensor([10])

# Inputs to the model
x = torch.randn(2, 3, 32, 32)
