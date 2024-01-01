
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
 
    def forward(self, x):
        # Split and then concatenate tensors in "channel" dimension
        v1, v2, v3 = torch.split(x, [-1, 1, -1], 1)
        y = torch.cat([v1, v3, v2], 1)
        return y

# Initializing a model
m = Model()

# Input to the model
x = torch.randn(1, 4, 64, 64)
