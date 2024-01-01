
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2):
        v1 = x2.view(x2.shape[1], -1)
        v2 = torch.cat((v1, v1), dim=1)
        v3 = torch.relu(v2)
        v4 = v1.view(v1.shape[0], -1)
        return torch.cat((v3, v3, v4), dim=1).view(-1)
# Inputs to the model
x2 = torch.randn(2, 3, 4)
