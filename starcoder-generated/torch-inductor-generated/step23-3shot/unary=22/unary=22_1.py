
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.act = torch.nn.Tanh()
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.act(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 10)
