
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.bmm(x2, v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 2)
x2 = torch.randn(2, 2)
