
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.squeeze(-1)
        x = torch.cat((x, x), dim=1)
        x = x.view(x.shape[0], x.shape[0], 1).repeat(1, 1, 2).flatten()
        return x
# Inputs to the model
x = torch.randn(2, 3, 1)
