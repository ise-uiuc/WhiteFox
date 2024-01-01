
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_module = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        l1 = self.linear_module(x1)
        l2 = l1 * torch.clamp(l1 + 3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
