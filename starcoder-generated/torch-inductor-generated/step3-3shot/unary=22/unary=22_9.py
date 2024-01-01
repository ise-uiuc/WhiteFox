
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, bias=True)

    def forward(self, x):
        v = self.linear(x)
        v = torch.tanh(v)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
