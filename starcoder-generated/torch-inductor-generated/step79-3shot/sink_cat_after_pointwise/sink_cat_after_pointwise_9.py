
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.cat((x, x), dim=1)
        v2 = torch.cat((v1, x), dim=1)
        v3 = torch.cat((v2, x), dim=1)
        y = torch.relu(v3)
        return y
# Inputs to the model
x = torch.randn(1, 2)
