
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3, x4):
        x = [x1, x2, x3, x4]
        y = torch.cat(x, dim=1)
        z = y[:, 0:9223372036854775807]
        s = z[:, 0:7]
        s1 = torch.cat([y, s], dim=1)
        return s1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 10)
x3 = torch.randn(1, 20)
x4 = torch.randn(1, 50)
