
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 8)
        self.fc2 = torch.nn.Linear(8, 8)
        self.fc3 = torch.nn.Linear(8, 1)
 
    def forward(self, x):
        w = torch.ones_like(self.fc2.weight)
        x = self.fc1(x)
        x = self.fc2(x) * 0.7978845608028654
        x = x + (self.fc2.weight * w * 0.044715)
        x = torch.tanh(x) * 0.5
        x = x + self.fc3(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 1)
