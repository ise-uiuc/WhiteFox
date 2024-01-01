
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(12, 8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.addmm(x1, v1, v1)
        v3 = torch.cat([v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12)
