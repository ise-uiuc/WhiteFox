
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        x = x + self.bias
        return x
# Inputs to the model
self.bias = torch.nn.Parameter(torch.rand_like(x))
