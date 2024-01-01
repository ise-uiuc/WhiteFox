
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = (x.view(int(x.shape[0]/2), -1), x.view(int(x.shape[0]/2), -1))
        x = x.permute(1, 0, 2).contiguous()
        y = x.view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(10, 16, 28, 28)
