
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 8)
 
    def forward(self, x):
        return torch.sigmoid(self.fc1(x))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
