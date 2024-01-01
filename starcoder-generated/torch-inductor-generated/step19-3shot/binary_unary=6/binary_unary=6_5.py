
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(64 * 64 * 3, 40)
 
    def forward(self, x):
        v0 = x.view(-1, 64 * 64 * 3)
        v1 = self.fc(v0)
        v2 = v1 - 2.78017653604e-06
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
