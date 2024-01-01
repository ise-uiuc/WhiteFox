
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 - other
        return  v2

# Initialize the model
m = Model()

# Inputs to the model
x = torch.randn(1, 64)
