
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(torch.clamp(l1 + 3, 0, 6), min=0, max=6) / 6
        return l2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
