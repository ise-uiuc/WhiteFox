
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = torch.addmm(x2, v1, v1)
        v3 = [v2]
        v4 = torch.cat(v3, dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(4, 3, 3)
