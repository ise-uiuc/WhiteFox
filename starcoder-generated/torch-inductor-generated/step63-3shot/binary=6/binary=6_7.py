
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 4)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = v2 * -0.5
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(2, 1)
