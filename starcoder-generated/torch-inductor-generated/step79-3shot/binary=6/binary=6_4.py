
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 - 0.1
        v3 = v2 * 0.7071067811865476
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
