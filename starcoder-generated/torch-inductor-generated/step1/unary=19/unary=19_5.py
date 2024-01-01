
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 10)
        self.fc2 = torch.nn.Linear(10, 5)
 
    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.sigmoid(self.fc1(x))))

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
