
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 6)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.tensor([3.4, 5.7, 2.9, 9.4, 6.0, 2.9])
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 8)
