
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(25, 30, bias=False)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + 0.01
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 25)
