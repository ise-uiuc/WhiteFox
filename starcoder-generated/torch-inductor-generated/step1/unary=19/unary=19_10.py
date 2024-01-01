
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
 
    def forward(self, x):
        v1 = torch.sigmoid(self.fc1(x))
        v2 = self.fc2(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 7)
