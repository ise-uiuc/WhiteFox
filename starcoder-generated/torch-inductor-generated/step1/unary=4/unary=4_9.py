
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 8)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = v1 * 0.5
        v3 = self.fc(x) * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
