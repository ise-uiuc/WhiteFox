
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 1, bias=True)
    
    def forward(self, x):
        t = self.fc1(x)
        return t

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
