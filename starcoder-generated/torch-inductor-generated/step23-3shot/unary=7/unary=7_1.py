
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def forward(self, x1, x2, x3):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(l1 + 3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16)
x2 = torch.randn(16)
x3 = torch.randn(16)
