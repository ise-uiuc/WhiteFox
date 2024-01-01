
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 256)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(torch.abs(l1), min=0, max=6) + 3
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
