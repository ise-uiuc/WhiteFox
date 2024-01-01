
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 10)
 
    def forward(self, x):
        v1 = self.fc(x)
        v2 = torch.gt(v1, 0)
        v3 = v1.where(v2, -0.2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
