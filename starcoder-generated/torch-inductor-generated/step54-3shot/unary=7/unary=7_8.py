
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(torch.nn.functional.relu6(l1 + 3), max=6)
        return l3 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
