
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(400, 6400)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + 1.0
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 400)
