
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)
 
    def forward(self, x1, **kwargs):
        x2 = self.linear(x1)
        x3 = x2 + kwargs["other"]
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
other = torch.randn(1, 5)
