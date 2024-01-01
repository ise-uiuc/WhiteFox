
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 4)
 
    def forward(self, x1, x2):
        v1 = self.fc(x1)
        v2 = v1 + x2
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 4)
