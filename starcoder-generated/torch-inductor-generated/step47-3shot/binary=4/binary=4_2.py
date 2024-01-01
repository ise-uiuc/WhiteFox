
class Model(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.linear = torch.nn.Linear(param[0], param[1])
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m = Model([128, 32])

# Inputs to the model
x1 = torch.randn(1, 128)
x2 = torch.randn(1, 32)
