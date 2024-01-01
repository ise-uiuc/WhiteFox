
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        __boolean_tensor__ = v1 > 0
        __negative_slope__ = 0.01
        v2 = v1 * __negative_slope__
        v3 = torch.where(__boolean_tensor__, v1, v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
