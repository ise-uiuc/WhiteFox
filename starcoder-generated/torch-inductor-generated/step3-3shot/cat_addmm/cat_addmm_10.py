
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(16, 8)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = torch.tanh(v1)
        v3 = torch.addmm(v2, x2, v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(8, 16)
