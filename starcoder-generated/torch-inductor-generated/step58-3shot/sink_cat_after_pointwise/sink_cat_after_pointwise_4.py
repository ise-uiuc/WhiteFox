
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x, x], dim=0)
        x = x.view(x.shape[0], -1).relu()
        return x[:, :x.shape[0]]
# Inputs to the model
x = torch.randn(2, 3, 4)
