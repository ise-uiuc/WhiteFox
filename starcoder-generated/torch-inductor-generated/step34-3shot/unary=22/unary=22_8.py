
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 4)
 
    def forward(self, x2):
        v2 = self.fc(x2)
        v4 = torch.tanh(v2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(3, 4)
