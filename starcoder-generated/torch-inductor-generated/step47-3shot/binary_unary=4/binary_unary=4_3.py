
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(300, 1000)
 
    def forward(self, x1, other=None):
        if other is None:
            other = torch.empty((1, 1000), dtype=torch.float32)
            nn.init.uniform_(other, a=-20, b=20)
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 300)
