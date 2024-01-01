
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, input):
        l1 = self.linear(input)
        l2 = l1 * torch.clamp(l1 + 3, min=0, max=6)
        out = l3 / 6
        return out

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
