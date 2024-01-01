
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3*8*8, 1)
 
    def forward(self, x):
        v1 = x.reshape(1, -1)
        v3 = self.fc(v1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1, 3, 64, 64)
