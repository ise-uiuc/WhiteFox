
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - x2 # x2 is the input tensor to subtract
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.tensor(1)
