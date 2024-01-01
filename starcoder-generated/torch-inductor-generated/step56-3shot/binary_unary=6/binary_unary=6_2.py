
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(128, 1)
 
    def forward(self, x1):
        v1 = self.linear0(x1)
        v2 = v1 - 2
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128)
