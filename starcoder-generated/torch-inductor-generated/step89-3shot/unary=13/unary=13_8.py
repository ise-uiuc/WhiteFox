
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear_1 = torch.nn.Linear(64,64)
 
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = torch.sigmoid(v1)
        return v1 * v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
__output_2__ = m(x1)

