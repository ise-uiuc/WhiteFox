
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(3072, 4096)
        self.fc1 = torch.nn.Linear(4096, 512)
 
    def forward(self, x):
        v1 = self.fc0(x)
        v2 = v1
        v3 = torch.tanh(v2)
        v4 = self.fc1(v3)
        v5 = v4
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(100, 3072)
