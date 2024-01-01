
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        m1 = self.linear(x1)
        m2 = m1 + 3
        m3 = torch.clamp_min(m2, 0)
        m4 = torch.clamp_max(m3, 6)
        m5 = m4 / 6
        return m5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
