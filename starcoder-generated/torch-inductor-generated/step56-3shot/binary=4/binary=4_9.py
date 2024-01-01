
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.other = torch.randn(8, 8)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Input to the model
x = torch.rand(1, 3)
