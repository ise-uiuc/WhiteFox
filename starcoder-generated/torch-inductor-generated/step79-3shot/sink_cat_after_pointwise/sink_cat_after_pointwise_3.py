
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v2 = torch.cat((x1, x2), dim=1)
        v1 = v2.tanh()
        y = torch.relu(v1)
        return y
# Inputs to the model
x1 = torch.randn(1, 2, 4)
x2 = torch.randn(1, 2, 4)
