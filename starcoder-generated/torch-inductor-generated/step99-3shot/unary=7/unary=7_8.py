
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        r = self.linear(x)
        m = r * torch.clamp(r + 3, 0, 6) / 6
        return m

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
