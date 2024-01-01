
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1024, 100)
 
    def forward(self, x2):
        v1 = self.fc(x2)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(2, 1024)
