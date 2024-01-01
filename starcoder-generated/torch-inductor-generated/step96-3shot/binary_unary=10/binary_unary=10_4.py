
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128 * 128, 10)
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + torch.ones(1,10, dtype=torch.float)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128 * 128)
