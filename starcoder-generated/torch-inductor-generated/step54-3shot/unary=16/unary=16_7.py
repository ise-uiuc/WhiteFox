
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(50, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.nn.functional.relu(v1)
        return (v2, v1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 50)
__output__, 