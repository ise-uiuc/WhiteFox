
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, torch.zeros([x.shape[0], 1])], dim=1)
        ret = y.reshape(y.shape[0], 1) if y.shape[0] > 1 else y
        return ret

# Inputs to the model
x = torch.randn(3)
