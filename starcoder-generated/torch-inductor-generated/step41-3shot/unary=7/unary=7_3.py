
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)

    def forward(self, inp):
        l1 = self.linear(inp)
        l2 = l1 * torch.clamp(l1 + 3, min=0, max=6.)
        l3 = l2 / 6.
        return l3
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
