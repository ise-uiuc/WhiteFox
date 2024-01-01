
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(100, 10)
 
    def forward(self, x0):
        v0 = x0.view(x0.size()[0], 100)
        v1 = self.fc(v0)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(2, 100)
