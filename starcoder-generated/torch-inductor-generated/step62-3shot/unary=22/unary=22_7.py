
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(28 * 28, 10)
 
    def forward(self, x1):
        v1 = torch.flatten(x1, 1)
        v2 = self.fc(v1)
        v3 = torch.tanh(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
