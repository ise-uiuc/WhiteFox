
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - 0.5
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
__input__ = torch.randn(1, 16)
