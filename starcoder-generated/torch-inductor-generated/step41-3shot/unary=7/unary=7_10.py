
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
 
    def forward(self, x):
        l1 = x.mean(-1).view(-1)
        l2 = m1 * torch.clamp_min_max(min=0, max=6.0, l1 + 3.0)
        l3 = l2 / 6.0
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16, 32)
