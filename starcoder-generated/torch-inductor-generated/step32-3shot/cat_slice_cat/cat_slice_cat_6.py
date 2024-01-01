
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, xlist1, xlist2, xlist3):
        v1 = torch.cat(xlist1, dim=1)
        v2 = v1[:, 0:(9223372036854775807)]
        v3 = v2[:, 0:(v1.shape[2] - 2)]
        v4 = torch.cat([v1, v3], dim=1)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
x2 = torch.randn(2, 32, 128)
x3 = torch.randn(1, 128, 128)
