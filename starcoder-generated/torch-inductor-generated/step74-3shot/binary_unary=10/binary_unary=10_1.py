
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(300, 600, bias=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v1)
        v4 = v3 + v1
        return v4

# Initializing the model
n = Model()

# Inputs to the model
x1 = torch.randn(1, 300)
