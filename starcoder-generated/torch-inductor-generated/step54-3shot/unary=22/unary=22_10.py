
class Model(torch.nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.linear = torch.nn.Linear(20, 5, bias)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
