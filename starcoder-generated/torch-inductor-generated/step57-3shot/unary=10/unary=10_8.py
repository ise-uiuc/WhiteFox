
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        b1 = self.linear(x1)
        b2 = b1 + 3
        b3 = torch.clamp_min(b2, 0)
        b4 = torch.clamp_max(b3, 6)
        b5 = b4 / 6
        return b5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
