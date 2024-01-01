
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 5)
 
    def forward(self, x):
        x1 = self.linear_1(x)
        x2 = self.linear_2(x)
        x3 = x1 + x2
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 5)
