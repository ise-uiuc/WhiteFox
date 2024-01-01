
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 8)
 
    def forward(self, x1, __other__):
        v1 = self.linear(x1, __other__=__other__)
        v2 = v1 + __other__
        v3 = v2.relu()
        return v3

# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(4, 32)
