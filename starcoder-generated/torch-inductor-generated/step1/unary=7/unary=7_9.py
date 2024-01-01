
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 32, bias=False)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = v1 + 3
        v3 = v2.clamp(0, 6)
        v4 = v3 * 6
        v5 = v4 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
