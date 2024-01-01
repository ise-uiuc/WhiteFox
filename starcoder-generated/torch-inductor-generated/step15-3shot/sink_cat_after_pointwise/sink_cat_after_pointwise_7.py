
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.cat((x, x), dim=1)
        v2 = torch.cat((v1, v1), dim=1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(2, 3, 4)
