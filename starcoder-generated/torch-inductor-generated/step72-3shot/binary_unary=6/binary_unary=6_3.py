
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(3, 8, True)

# Inputs to the model
x1 = torch.randn(1, 3)
