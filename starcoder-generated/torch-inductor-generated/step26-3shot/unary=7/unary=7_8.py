
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x1):
        u1 = self.linear(x1)
        u2 = u1 * torch.clamp(torch.abs(u1) + 3, 0, 6)
        u3 = u2 / 6
        return u3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
