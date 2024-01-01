
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat((x1, x2), dim=2)
        v3 = torch.cat((x2, x2), dim=1)
        v4 = torch.cat((v1, v3), dim=1)
        v5 = torch.cat((v4, v3), dim=1)
        v6 = torch.cat((v1, v5), dim=1)
        v2 = torch.relu(v6)
        v7 = v2.view(-1)
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 3)
x2 = torch.randn(1, 3)
