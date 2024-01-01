
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        v3 = relu(v2)
        return v3

# Model inputs
x1 = torch.randn(1, 8)

# Initializing the model
torch.manual_seed(0)
m = Model()
