
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x[:, :, None]
        y = y.squeeze(dim=3)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
