
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat((x1, x1), dim=1)
        v2 = torch.cat((v1, v1), dim=1)
        v3 = torch.relu(v2)
        v3.view(-1)
        return v3.view(-1)
# Inputs to the model
x1 = torch.randn(1, 2)
