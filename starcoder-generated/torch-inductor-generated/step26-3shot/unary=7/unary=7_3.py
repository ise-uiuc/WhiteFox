
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
 
    def forward(self, x):
        l1 = self.linear(x)
        l2 = l1 * torch.clamp(torch.clamp(l1 + 3, min=0), max=6) * 6
        return l2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
