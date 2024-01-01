
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
