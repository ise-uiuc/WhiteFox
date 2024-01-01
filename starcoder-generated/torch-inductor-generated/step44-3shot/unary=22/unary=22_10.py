
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 12)
 
    def forward(self, x1):
        hidden = self.linear(x1)
        out = torch.tanh(hidden)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)
