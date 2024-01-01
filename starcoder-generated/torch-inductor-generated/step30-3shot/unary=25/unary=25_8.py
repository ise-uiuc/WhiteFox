
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = -1 * 0.01
        v4 = torch.where(v2, v1, v3)
        return 

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(7, 3)
