
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 1)
 
    def forward(self, x2):
        v1 = self.fc(x2)
        v2 = v1 - 1.0
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 16)
self.other = torch.tensor(2.)
