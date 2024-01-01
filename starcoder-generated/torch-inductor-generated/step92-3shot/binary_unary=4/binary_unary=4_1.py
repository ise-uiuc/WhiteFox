
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.fc = torch.nn.Linear(32, 8)
        self.other = other
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model with parameter `other`
m = Model(torch.randn(1, 8))

# Inputs to the model
x1 = torch.randn(1, 32)
