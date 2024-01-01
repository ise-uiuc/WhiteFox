
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3, bias=False)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1.clamp(min=0, max=6)
        v3 = v2.add(3)
        v4 = v3 * 1./6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
