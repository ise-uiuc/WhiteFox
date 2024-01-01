
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(16, 32)
        self.l2 = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        l1 = self.l1(x1)
        l2 = l1 * torch.clamp(min=0, max=6, l1 + 3)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 16)
