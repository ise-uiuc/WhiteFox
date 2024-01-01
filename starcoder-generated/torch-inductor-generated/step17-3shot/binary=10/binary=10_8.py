
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=3, out_features=4, bias=True)
 
    def forward(self, x1, x2=None):
        v = self.linear(x1)
        if x2 is None:
            return v
        else:
            return v + x2
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
