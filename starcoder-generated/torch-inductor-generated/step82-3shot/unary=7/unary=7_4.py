
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        m1 = torch.nn.ReLU() # ReLU
        v1 = m1(v1)
        v3 = v1 / 6
        v2 = torch.clamp(v3, 0, 6) + 3
        v4 = v2 * v1
        return v4

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 3, 1, 1)
