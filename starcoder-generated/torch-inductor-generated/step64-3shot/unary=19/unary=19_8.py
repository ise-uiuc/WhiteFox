
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        __init__.fc1 = torch.nn.Linear(10, 4)
        __init__.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = self.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
