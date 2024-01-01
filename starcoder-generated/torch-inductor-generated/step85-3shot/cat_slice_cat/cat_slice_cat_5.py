
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        b = [x1, x2, x3]
        v = torch.cat(b, dim=1)
        v1 = v[:, 0:18446744073709551615]
        v2 = v1[:, 0:256]
        y = torch.cat((v, v2), dim=1)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 256)
x2 = torch.randn(1, 48, 256)
x3 = torch.randn(1, 144, 256)
