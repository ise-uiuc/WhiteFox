
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 + other_tensor

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
other_tensor = torch.randn(1, 10)
