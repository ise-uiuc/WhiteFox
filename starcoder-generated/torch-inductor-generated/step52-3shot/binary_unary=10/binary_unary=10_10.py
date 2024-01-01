
class Model(torch.nn.Module):
    def __init__(self, in_shape, out_channels):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(in_shape, out_channels)
 
    def forward(self, x):
        return relu(self.linear(x) + x)

# Initializing the model
m = Model(4, 8)

# Inputs to the model
x1 = torch.randn(1, 4)
