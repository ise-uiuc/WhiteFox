
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(128, 32, bias=False)
        self.other = torch.tensor([-1], dtype=torch.float32)
    
    def forward(self, x):
        v1 = x.mean([2, 3])
        v2 = self.fc(v1)
        v3 = v2 + self.other
        v4 = v3.relu()
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 128, 16, 16)
