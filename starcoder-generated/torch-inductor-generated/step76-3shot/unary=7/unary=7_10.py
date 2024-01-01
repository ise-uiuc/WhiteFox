
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 6, True)
 
    def forward(self, x0):
        l1 = self.linear(x0)
        l2 = l1 * torch.clamp(l1 + 3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 10)
