
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Flatten()
        self.fc2 = torch.nn.Linear(64*64+4,1)
 
    def forward(self, x1, other):
        v1 = self.fc(x1)
        v2 = torch.cat([v1, other],dim=1)
        v3 = self.fc2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
x2 = torch.randn(1, 4)
