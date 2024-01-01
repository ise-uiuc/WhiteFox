
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        y1 = torch.cat([x1, x2, x3], dim=1)
        y2 = y1[:, 0:9223372036854775807]
        y3 = y2[:, 0:int(y2.shape[1] / 2)]
        y4 = torch.cat([y1, y3], dim=1)
        return y4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 9223372036854775807, 7)
x2 = torch.randn(1, 9223372036854775807, 7)
x3 = torch.randn(1, 9223372036854775807, 7)
