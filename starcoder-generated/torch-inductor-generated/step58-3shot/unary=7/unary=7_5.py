
class Model(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.linear = nn.Linear(__IN_SIZE__, 10, bias)
        self.bias = bias
    def forward(self, x):
        if self.bias:
            l1 = self.linear(x)
        else:
            l1 = self.linear(x) + 1
        l2 = l1 * torch.clamp(l1, max=6.0) + 3
        l3 = l2 / 6.0
        return l3

# Initializing the model
m1 = Model(bias=True)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
