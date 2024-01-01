
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bias = torch.nn.Parameter(data=torch.zeros(8, 64, 64), requires_grad=True)
 
    def forward(self, x):
        x1 = x + self.bias
        x2 = self.conv(x1)
        x3 = torch.relu(x2, 0.1)
        return x3

# Initializing the model
m = Model()
print(m.conv.weight.sum(), m.bias.sum())

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
