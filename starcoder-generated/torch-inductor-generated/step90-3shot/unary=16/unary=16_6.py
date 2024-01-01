
class Model(torch.nn.Module):
    def __init__(self, dinp, dout):
        super().__init__()
        self.linear = torch.nn.Linear(dinp, dout)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model(10, 64)

# Inputs to the model
x1 = torch.randn(1, 10)
