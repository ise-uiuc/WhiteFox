
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20)
 
    def forward(self, x):
        v1 = self.fc(x)
        v1 = v1 + 1.37
        v1 = torch.relu(v1)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 16)
