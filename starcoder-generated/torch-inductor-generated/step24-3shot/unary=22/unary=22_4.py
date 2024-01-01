
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 1024)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(5, 1024)
