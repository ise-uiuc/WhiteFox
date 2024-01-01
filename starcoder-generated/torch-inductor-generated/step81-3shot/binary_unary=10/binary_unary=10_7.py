
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        return self.linear(x1) + other_tensor

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
