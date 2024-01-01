
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=1)
        x = torch.cat((x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=1)
        x = torch.cat((x, x), dim=1)
        x = torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 2)
