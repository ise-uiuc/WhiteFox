
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        x3 = [x1, x2]
        x4 = torch.cat(x3, dim=1)
        x5 = x4[:, 0:9223372036854775807]
        x6 = x5[:, 0:size]
        x7 = torch.cat([x4, x6], dim=1)
        return x7

# Initializing the model
m = Model()

# Inputs to the model
shape = (1, 3, 14, 14)
size = shape[1]
x1 = torch.randn(shape)
x2 = torch.randn(shape)
