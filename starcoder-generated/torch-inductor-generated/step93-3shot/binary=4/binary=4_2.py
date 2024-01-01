
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 4)
 
    def forward(self, x1, other):
        v1 = torch.matmul(
            x1,
            self.linear1.weight,
        )
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
__other__ = torch.randn(1, 4)
