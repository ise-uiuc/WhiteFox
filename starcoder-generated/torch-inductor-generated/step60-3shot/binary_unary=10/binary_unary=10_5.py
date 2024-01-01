
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linearLayer = torch.nn.Linear(320, 2)
 
    def forward(self, x2):
        v1 = self.linearLayer(x2)
        v2 = v1 + torch.rand(1, 2)
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 2, 5, 5)
