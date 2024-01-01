
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x, x2):
        v1 = self.linear(x)
        v2 = torch.tanh(v1)
        v3 = v2 * x2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(128, 128)
x2 = torch.randn(128, 128)
