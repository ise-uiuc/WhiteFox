
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)
 
    def forward(self, x1, x2):
        v1 = self.linear(x2)
        v2 = v1 - x1
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model(3, 8)

# Inputs to the model
x1 = torch.randn(3)
x2 = torch.randn(1, 3, 64, 64)
