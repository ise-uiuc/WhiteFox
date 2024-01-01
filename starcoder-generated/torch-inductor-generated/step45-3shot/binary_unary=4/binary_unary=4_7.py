
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 1000)
 
    def forward(self, x1, y1=4):
        v1 = self.linear(x1)
        v2 = v1 + y1
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224)
