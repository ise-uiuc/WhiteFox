
class Model(torch.nn.Module):
    def __init__(self, weight):
        super(Model, self).__init__()
        self.weight = weight

    def forward(self, x):
        out = torch.mm(x, self.weight)
        return out

weight = torch.randn(1000, 1000) # weight should be initialized as a 1000-by-1000 matrix
# Inputs to the model
x = torch.randn(100, 100)
