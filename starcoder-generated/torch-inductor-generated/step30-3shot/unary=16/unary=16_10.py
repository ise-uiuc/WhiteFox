
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc3 = torch.nn.Linear(28*28, 30)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x2):
        v1 = self.fc3(x2)
        v2 = self.relu(v1)
        return v2


# Initializing the model
m = Model()

# Inputs to the model
x1, x2 = torch.randn(1, 784)
