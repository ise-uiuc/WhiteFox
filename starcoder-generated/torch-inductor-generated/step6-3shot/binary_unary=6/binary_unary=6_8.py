
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 100)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - y
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 512)
y = torch.randn(1, 512)
v1 = torch.randn(1, 512)
