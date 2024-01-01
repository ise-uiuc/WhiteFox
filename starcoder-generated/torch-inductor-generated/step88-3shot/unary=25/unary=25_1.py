
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.gt(v1, 0)
        v3 = v1 * 0.03
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 2)
