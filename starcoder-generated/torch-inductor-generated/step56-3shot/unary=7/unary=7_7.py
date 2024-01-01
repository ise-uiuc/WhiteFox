
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(torch.clamp(l1 + 3, max=6, min=0), max=6, min=0)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
