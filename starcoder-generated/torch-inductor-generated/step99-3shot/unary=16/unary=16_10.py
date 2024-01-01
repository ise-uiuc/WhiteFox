
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x2):
        v7 = self.linear(x2)
        v8 = F.relu(v7)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64)
