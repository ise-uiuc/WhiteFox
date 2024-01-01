
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x[:, 0:1], x[:, 1:4]], dim=1)
        if y.dim() > 2:
            y = y.view(-1)
        y = torch.pow(y, 2)
        return y.unsqueeze(1)
# Inputs to the model
x = torch.randn(2, 1, 4)
