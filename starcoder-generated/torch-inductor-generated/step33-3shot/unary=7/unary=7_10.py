
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(3, 64)
 
    def forward(self, x2):
        l1 = self.l(x2)
        l2 = l1 * torch.clamp(torch.add(l1, 3), min=0, max=6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
