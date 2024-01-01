
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 32)
        self.linear2 = torch.nn.Linear(32, 5)
 
    def forward(self, x, other):
        v1 = self.linear1(x)
        v2 = v1 + other
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Initialize the other tensor specified by the "other" keyword argument (this tensor is initialized with random values)
other = 2 * torch.rand(1, 5)

# Inputs to the model
x = torch.randn(1, 10)
