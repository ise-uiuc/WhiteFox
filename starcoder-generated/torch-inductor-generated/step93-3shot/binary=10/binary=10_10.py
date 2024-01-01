
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_trans = torch.nn.Linear(5, 10)
 
    def forward(self, x1):
        v1 = self.linear_trans(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
