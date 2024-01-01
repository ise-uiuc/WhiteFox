
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mlp = torch.nn.Linear(224 * 224, 4096)
 
    def forward(self, x1):
        v1 = self.mlp(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224 * 224)
