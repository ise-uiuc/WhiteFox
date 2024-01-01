
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1):
        v1 = torch.addmm(x1, torch.rand(64, 64), torch.rand(64, 64))
        return [v1]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(64, 50, 20)
