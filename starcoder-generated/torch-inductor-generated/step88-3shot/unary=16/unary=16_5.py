
class Model(torch.nn.Module):
    def __init__(self):
         super().__init__()
         self.fc = torch.nn.Linear(6, 8)
 
    def forward(self, x1):
        r1 = self.fc(x1)
        r2 = torch.relu(r1)
        return r2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
