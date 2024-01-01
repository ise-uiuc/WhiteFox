
class Model(torch.nn.Module):
    def __init__():
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
 
    def forward(self, x1, x3):
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x3 = torch.randn(1, 3)
