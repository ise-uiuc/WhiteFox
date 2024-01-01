
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat((x1, x1), dim=1)
        v2 = v1.tanh()
        y = v2.sigmoid()
        return y
# Inputs to the model
x1 = torch.randn(1, 16, 16, 16, 16)
