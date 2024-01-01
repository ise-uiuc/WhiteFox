
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(13, 7, bias=False)
 
    def forward(self, x, other=None):
        v1 = self.linear(x)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn((6, 13), requires_grad=False)
other = torch.randn((6, 7), requires_grad=False)
