
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        x2 = torch.cat([x1, x1], dim = 1)
        y1 = x2[:, 0:9223372036854775807]
        y2 = y1[:, 0:-1]
        y3 = torch.cat([x2, y2], dim = 1)
        return y3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 1)
