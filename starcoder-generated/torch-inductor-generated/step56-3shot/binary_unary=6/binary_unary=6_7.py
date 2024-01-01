
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1000, 2048)
 
    def forward(self, x1):
        v1 = self.layer1(x1)
        v2 = v1 - 0.1
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000)
