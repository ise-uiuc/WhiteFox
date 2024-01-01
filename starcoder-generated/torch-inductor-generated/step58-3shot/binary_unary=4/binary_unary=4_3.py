
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1, __param_other=None):
        v1 = self.linear(x1)
        v2 = v1 + __param_other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 3)
