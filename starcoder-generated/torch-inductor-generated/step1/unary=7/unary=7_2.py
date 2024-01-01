
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 5)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.add(v1, 3)
        v3 = torch.nn.functional.relu6(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
