
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.cat((x, x), 1)
        v2 = torch.cat((v1, x), dim=-1)
        return v2.view(v2.size(0), v2.size(1), 1)
# Inputs to the model
x = torch.randn(3, 4, 2)
