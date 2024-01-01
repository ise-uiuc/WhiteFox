
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        v1 = linear(x1)
        v2 = F.relu(v1)
        return v2

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(10, 10)
