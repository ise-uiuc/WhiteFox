
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 16)
 
    def forward(self, x2):
        y2 = self.linear(x2)
        a2 = torch.clamp(y2, min=0, max=6)
        y3 = y2 + a2
        y4 = y3 / 6
        return y4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 12)
y = m(x2)

