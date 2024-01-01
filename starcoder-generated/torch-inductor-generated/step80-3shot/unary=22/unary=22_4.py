
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 32)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.tanh(v1)
        v3 = v1 * 0.7
        v4 = v3 + 1
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
