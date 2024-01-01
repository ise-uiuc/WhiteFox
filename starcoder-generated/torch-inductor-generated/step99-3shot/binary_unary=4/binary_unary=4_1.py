
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64*64, 8192)
 
    def forward(self, x1):
        m1 = self.linear(x1)
        m2 = m1 + x2
        m3 = torch.relu(m2)
        return m3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 64, 64)
x2 = torch.randn(128, 8192)
