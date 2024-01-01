
class Model(torch.nn.Module):
    def __init__(self, W, b):
        super().init__()
        self.linear = torch.nn.Linear(4, W, b)
 
    def forward(self, x):
        l1 = self.linear(x)
        l2 = l1 * torch.clamp(l1 + 3, min=0, max=6)
        l3 = l2 / 6
        return l3

# Initializing parameters of the model
W = torch.randn(9)
b = torch.randn(9)
