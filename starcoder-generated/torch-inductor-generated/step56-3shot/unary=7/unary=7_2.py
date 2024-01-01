
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256, bias=False)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * (l1.clamp(min=0, max=6) + 3)
        return l2 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 2, 8, 12)
