
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(12, 4)
 
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 12)
