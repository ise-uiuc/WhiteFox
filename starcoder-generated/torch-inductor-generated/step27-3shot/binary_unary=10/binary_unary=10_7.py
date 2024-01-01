
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 1)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + x
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(64)
