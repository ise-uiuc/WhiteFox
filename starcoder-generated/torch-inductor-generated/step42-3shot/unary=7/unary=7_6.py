
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 3
        m = torch.nn.ReLU6() # ReLU6 is defined in PyTorch
        v3 = m(v2)
        v4 = v1 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
