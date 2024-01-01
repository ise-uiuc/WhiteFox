
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], x.numel() // x.shape[0])
        return x.sigmoid().view(x.shape[0], x.shape[1], 1)
# Inputs to the model
x = torch.randn(2, 2, 2)
