
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x2):
        v9 = self.linear(x2)
        v10 = v9 - 6
        v11 = relu(v10)
        return v11

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 10)
