
class Model(torch.nn.Module):
    def __init__(self, in_channel=3, in_size=64):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(in_channel, 1)
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
