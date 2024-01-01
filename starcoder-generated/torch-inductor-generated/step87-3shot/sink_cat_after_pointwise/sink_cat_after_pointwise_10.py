
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, w, b):
        x = torch.cat((x, x, x), dim=1)
        x = x.view(x.shape[0], x.shape[2])
        x = x + b
        ret = torch.matmul(torch.tanh(x), w)
        return ret
# Inputs to the model
x = torch.randn(5, 2, 3)
w = torch.randn(6, 3)
b = torch.randn(6, 1)
