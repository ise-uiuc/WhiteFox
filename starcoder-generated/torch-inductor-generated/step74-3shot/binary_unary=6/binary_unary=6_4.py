
class Model(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.linear = torch.nn.Linear(20, out_channels, bias=False)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 2.0
        return F.relu(v2)

# Initializing the model
m = Model(4)

# Inputs to the model
x1 = torch.randn(1, 20)
