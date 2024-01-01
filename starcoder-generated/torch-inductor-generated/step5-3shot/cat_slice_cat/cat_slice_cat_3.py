
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:5]
        v3 = v2[:, 1:3]
        v4 = v1[:, 2:37]
        return v3, v4
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 5, 20, 2)
x2 = torch.randn(1, 36, 2, 5)
y1, y2 = m(x1, x2)

