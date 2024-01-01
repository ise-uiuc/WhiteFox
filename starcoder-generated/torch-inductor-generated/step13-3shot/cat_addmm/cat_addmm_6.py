
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(6, 10)
 
    def forward(self, x1):
        v1 = x1.view(-1, 6)
        v2 = torch.addmm(v1, self.fc.weight, self.fc.bias.view(1, -1))
        v3 = torch.cat([v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6, 10, 10)
