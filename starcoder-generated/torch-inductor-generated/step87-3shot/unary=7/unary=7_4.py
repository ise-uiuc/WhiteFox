
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1, x2, x3):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.tanh(self.linear(x2)) + 3, 0, 6)
        v3 = v2 / 6
        v4 = self.linear(x3)
        v5 = v4 / 6
        v6 = v3 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
x2 = torch.randn(2, 10)
x3 = torch.randn(2, 10)
