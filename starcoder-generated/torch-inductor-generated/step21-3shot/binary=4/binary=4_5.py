
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.fc = torch.nn.Linear(32, 64)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        return v1 + other

# Initializing the model
m = Model(torch.randn(1, 32))

# Inputs to the model
x1 = torch.randn(1, 32)
