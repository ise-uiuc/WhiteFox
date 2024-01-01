
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        m1 = self.linear(x1)
        m2 = m1 + 3
        m3 = torch.clamp(m2, 0, 6)
        m4 = m3 / 6
        return m4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
