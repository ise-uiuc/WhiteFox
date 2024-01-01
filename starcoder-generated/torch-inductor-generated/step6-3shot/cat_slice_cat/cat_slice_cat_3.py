
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x0, x1, x2):
        v0 = torch.cat([x0, x1, x2], dim=1)
        v1 = v0[:, 0:32767]
        v2 = v0[:, 0:v1.shape[1]]
        v3 = torch.cat([v0, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 2, 11, 23)
x1 = torch.randn(1, 19, 13, 29)
x2 = torch.randn(1, 8, 11, 12)
x3 = torch.randn(1, 25, 17, 23)
x4 = torch.randn(1, 31, 21, 22)
x5 = torch.randn(1, 10, 19, 19)
x6 = torch.randn(1, 16, 16, 15)
x7 = torch.randn(1, 23, 13, 27)
