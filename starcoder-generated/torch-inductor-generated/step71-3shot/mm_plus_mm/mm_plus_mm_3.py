
class Model(nn.Module):
    def __init__(self, linear):
        super(Model, self).__init__()
        self.linear = linear
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.linear(x)
        return x
# Inputs to the model
N, D_in, H, D_out = 64, 784, 200, 10
x = torch.randn(N, D_in)
linear = torch.nn.Linear(D_in, H, bias = False)
model = Model(linear)
y = model(x)
# Model endds

# Model begins
class Model(nn.Module):
    def __init__(self, linear):
        super(Model, self).__init__()
        self.linear = linear
    def forward(self, x):
        x = F.relu(self.linear(x))
        return self.linear(x) + x
# Inputs to the model
model = Model(linear)
x = torch.randn(N, D_in)
y = model(x)
# Model endds