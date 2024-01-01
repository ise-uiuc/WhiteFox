
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 13)
 
    def forward(self, x1):
        m1 = self.linear(x1)
        m2 = torch.clamp_min(m1, 1.78139)
        m3 = torch.clamp_max(m2, 0.30158)
        return m3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)
x2 = torch.randn(1, 5, 12)
