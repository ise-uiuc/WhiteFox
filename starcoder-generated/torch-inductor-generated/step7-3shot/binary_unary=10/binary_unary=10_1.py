
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + 1
        v3 = v2.nn.ReLU()
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
