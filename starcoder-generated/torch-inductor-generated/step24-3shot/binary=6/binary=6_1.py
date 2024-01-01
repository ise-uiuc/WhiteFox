
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 256, bias=True)
 
    def forward(self, x1):
        v1 = 0.7071067811865476
        v2 = self.fc(x1)
        v3 = 0.5 * v2
        v4 = v3 + v1
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 128)
