
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = torch.nn.Linear(10, 10)
        m.weight.data.fill_(1.0)
        self.linear = torch.nn.utils.weight_norm(m, dim=0)
 
    def forward(self, x1, other):
        v2 = self.linear(x1)
        v3 = v2 - other
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 10)
other = torch.randn(1)
