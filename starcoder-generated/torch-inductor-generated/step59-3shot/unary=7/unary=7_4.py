
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1):
        n1 = self.linear(x1)
        n2 = n1 * torch.clamp(n1 + 3, min=0, max=6)
        n3 = n2 / 6
        return n3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
