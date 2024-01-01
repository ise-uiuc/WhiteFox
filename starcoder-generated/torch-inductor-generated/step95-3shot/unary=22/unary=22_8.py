
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3072, 4096)
 
    def forward(self, x1):
        x2 = self.fc1(x1)
        x3 = torch.tanh(x2)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3072)
