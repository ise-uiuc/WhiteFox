
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 8)
 
    def forward(self, x1):
        v1 = torch.flatten(x1, 1)
        v2 = torch.addmm(v1, v1, v1)
        v3 = torch.unsqueeze(v2, 0)
        v4 = torch.cat([v1, v2, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1024, 1024)
