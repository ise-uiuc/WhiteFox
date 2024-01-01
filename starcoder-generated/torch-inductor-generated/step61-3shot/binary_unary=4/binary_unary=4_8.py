
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
 
        if other is not None:
            v1 += other

        v2 = torch.nn.functional.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 64)
x2 = torch.randn(128, 64)
