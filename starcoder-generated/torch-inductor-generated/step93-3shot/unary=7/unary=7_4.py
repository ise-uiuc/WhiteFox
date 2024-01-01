
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10, bias=True)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1.clamp(0, 6) + 3
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.ones(2, 10)
