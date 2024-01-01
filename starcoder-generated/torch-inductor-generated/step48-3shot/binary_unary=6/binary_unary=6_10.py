
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(16, 32)
  
    def forward(self, x1):
        v1 = self.fc0(x1)
        v2 = v1 - other
        v3 = torch.nn.functional.relu(v2)
        v4 = torch.nn.functional.max_pool2d(v3, 64, 64)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 16)
