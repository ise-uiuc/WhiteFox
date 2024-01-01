
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - (456.0 - (34.5 + (-13.9 * 0.5)))
        return v2

# Initializing the model
m = Model()
m.linear.weight.data = torch.tensor((((456.0 - (34.5 + (-13.9 * 0.5))),),))
m.linear.bias.data = torch.tensor(((-13.9 * 0.5,),))

# Inputs to the model
x1 = torch.randn(1, 1)
