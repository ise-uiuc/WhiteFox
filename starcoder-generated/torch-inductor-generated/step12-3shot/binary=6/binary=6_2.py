
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(in_features=128, out_features=64)
 
    def forward(self, x, other):
        v1 = self.linear_1(x)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(64, 128)
other = torch.randn(64,1)
