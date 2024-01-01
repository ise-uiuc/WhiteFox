
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(32, 1)
 
    def forward(self, t):
        t = torch.tanh(t)
        y = t.view(-1, 32)
        y = self.fc(y)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 32)
