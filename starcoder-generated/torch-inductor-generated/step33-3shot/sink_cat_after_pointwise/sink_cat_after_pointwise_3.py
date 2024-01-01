
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0):
        x1 = torch.transpose(x0, 0, -1).reshape(x0.shape[1], -1)
        x2 = torch.cat((x1, x0), dim=0)
        return x2
# Inputs to the model
x = torch.randn(2, 2, 2)
