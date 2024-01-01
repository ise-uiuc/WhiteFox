
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 256)
 
    def forward(self, x1):
        b1 = self.linear(x1)
        b2 = b1 * torch.clamp(b1 + 3, min=0, max=6)
        b3 = b2 / 6
        return b3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
