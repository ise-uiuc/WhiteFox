
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Linear(4 * 224 * 224, 32)
    
    def forward(self, x1):
        v1 = self.f(x1)
        v2 = v1 - 12.92
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4 * 224 * 224)
