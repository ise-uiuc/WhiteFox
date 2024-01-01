
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, __x__):
        l1 = self.linear(__x__)
        l2 = l1 * torch.clamp(torch.min(l1) + 3, min=0, max=6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
__x__ = torch.randn(4, 16)
