
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.dense(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
