
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(100, 100)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v3 = v1 - x2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
x2 = torch.randn(1, 100, 200)
