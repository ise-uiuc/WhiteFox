
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(3, 4)
        self.weight = torch.randn(4, 3, dtype=torch.float32)
        self.bias = torch.randn(4, dtype=torch.float32)
 
    def forward(self, x1, other=None):
        v1 = self.dense(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3
 
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 2, 3, dtype=torch.float32)
x2 = torch.randn(2, 4, dtype=torch.float32)
