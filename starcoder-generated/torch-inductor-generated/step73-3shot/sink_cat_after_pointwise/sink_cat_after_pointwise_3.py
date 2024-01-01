
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        if y.dim() == 1 or y.shape[0] == 1:
            y = y * x.shape[1]
        else:
            y = y.unsqueeze(-1).expand_as(x)
        y = torch.zeros_like(y)
        y = torch.randn_like(x.shape)
        return x + y.matmul(y)
# Inputs to the model
x = torch.randn(2, 3, 4)
