
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        return v1 - 0.5 # Subtract 0.5 from the output of the linear transformation

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
