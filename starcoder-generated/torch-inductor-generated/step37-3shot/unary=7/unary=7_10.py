
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(int(16 * 56 * 56), int(16 * 56 * 56))
       
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=-6, max=6)
        return v2 / 6.

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16 * 56 * 56)
