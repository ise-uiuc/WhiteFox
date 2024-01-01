
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2):
        v1 = x1.reshape((1, 2, 2))
        v2 = x2.reshape((1, 2, 2))
        v3 = torch.matmul(v1.permute(0, 2, 1), v2)
        return v3.reshape((2, 2))
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
