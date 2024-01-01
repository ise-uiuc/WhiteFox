
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, bias=True)
 
    def forward(self, __input__):
        l1 = self.linear(__input__)
        l2 = l1 * torch.clamp(l1 + 3, min=0, max=6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
