
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = x.view(2, -1)
        x = x.relu()
        x = x.view(2, -1)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
