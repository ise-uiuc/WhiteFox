
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.reshape(-1, 1, 2, 3).transpose(2, 3)
        y = torch.cat([y, y, y], dim=1)
        return y.reshape(y.shape[0], -1).relu()
# Inputs to the model
x = torch.randn(2, 1, 2, 3)
