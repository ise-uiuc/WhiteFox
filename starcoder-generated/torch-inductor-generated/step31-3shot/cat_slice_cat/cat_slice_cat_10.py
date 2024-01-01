
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, size):
        v1 = torch.cat([x1, x1, x1],dim=1)
        v2 = v1[:, 0:10000000000000]
        v3 = v2[:, 0:size]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
print("Input size of the convolution: {}".format(x1.shape))
x2 = 27
