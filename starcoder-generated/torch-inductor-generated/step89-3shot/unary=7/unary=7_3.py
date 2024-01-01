
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, l):
        l1 = self.linear(x)
        l2 = l1 * l1.clamp(min=0, max=6) + 3
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
l = torch.randn(100, 100)
