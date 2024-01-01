
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul = torch.nn.Linear(64, 128)
 
    def forward(self, x1, x2):
        v1 = self.matmul(x1)
        v2 = torch.cat([v1, x2], 1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 20)
x2 = torch.randn(1, 10, 10)
