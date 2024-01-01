
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 12)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1 + 3
        v3 = torch.nn.functional.relu6(v2)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
