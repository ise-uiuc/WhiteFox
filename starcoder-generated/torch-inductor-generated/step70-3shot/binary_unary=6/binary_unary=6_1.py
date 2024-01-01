
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 64, bias=True)
 
    def forward(self, x1):
        v1 = self.tanh(self.fc(x1))
        v2 = v1 - 0.5
        v3 = (v2 + 2) / 5
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
