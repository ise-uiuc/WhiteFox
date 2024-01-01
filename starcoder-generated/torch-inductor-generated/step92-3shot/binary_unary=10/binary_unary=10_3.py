
class Model(torch.nn.Module):
    def __init__(self, in_channels__linear, out_channels__linear, in_channels__add, out_channels__add):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels__linear, out_channels__linear)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        return v3

# Initializing the model
in_channels__linear = 31337
out_channels__linear = 123
m = Model(in_channels__linear, out_channels__linear)

# Inputs to the model
x1 = torch.randn(1, in_channels__linear, 100, 100)
x2 = torch.randn(1, out_channels__linear, 100, 100)
