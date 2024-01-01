
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1, x2):
        t1 = self.linear(x1)
        return t1 - x2

# Initializing the model
m = Model()

# Input_1 to the model
x1 = torch.randn(20, 10)

# Input_2 to the model
x2 = torch.randn(20, 10)

# Outputs of the model (20 x 10)
