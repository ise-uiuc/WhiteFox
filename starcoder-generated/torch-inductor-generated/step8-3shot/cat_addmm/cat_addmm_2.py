
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(144, 10,'relu')
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 * v1
        v3 = torch.sum(torch.abs(v2), dim=1, keepdim=True)
        v3 = v3 * 0.5
        v4 = v1 * v3
        v = torch.cat([v4], dim=1)
        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 144)
