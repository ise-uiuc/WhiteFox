
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2):
        v1 = x1.mm(x2)
        v2 = torch.stack((v1, v1, v1, v1, v1), dim=1)
        v3 = torch.stack((v2[0], v2[1], v2[2], v2[3], v2[4]), dim=0)
        v4 = v3.flatten(start_dim=1)
        return v4
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 3)
