
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 64)
 
    def forward(self, x1):
        v1 = x1.view(x1.size(0), -1)
        v2 = self.fc(v1)
        v3 = v2 - torch.tensor(4)
        v4 = torch.nn.ReLU(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
