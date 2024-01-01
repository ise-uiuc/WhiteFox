
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 32)
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = torch.tanh(v1)
        return v1 + v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 8)
