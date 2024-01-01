
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(100, 100)
 
    def forward(self, x, inp=1):
        v1 = self.fc(x)
        v2 = torch.mm(v1, v1)
        return v2 + inp

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 100)
