
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.tanh = torch.nn.Tanh()
        self.erf = torch.erf

    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * self.tanh(l1 + 3)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(18, 10)
