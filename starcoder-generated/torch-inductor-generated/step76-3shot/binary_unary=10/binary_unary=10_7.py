
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(30, 100)
        self.relu = torch.nn.ReLU()
 
    def forward(self, x):
        x1 = self.fc0(x)
        x2 = x1 + x
        x3 = self.relu(x2)
        return x3
 
# Initializing the model
m = Model()
 
# Inputs to the model
x = torch.randn(1, 30)
