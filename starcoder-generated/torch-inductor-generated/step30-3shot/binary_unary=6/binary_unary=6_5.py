
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 32)
        self.other = torch.full((32, ), -2)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - self.other
        v3 = v2.clamp(min=0)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
