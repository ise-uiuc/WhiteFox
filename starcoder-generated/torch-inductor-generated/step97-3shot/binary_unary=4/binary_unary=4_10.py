
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1, other_tensor):
        v1 = self.linear(x1)
        v2 = v1 + other_tensor
        v3 = v2 + 1
        v4 = torch.sin(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
