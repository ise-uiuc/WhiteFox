
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat((x1, x1), dim=1)
        v2 = v1.view(2, -1, 1)
        v3 = v2.squeeze(dim=0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 1)
