
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 8, 3)
        self.linear2 = torch.nn.Linear(3, 8, 3)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
