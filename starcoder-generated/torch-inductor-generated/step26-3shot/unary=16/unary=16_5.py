
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
    v1 = linear(x1)
    v2 = torch.tanh(v1)
    return torch.cat((v2, torch.neg(v2)))

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
