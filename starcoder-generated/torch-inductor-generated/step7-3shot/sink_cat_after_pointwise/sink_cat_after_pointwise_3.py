
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat((x1.view(-1), x2.view(-1)), dim=1)
        if v1.dim() == 1 or v1.shape[0] == 1:
            y = v1.relu()
        else:
            v1 = v1.view(v1.shape[0], -1)
            y = v1.relu()
            y = y.view(v1.shape[0], -1).relu()
        return y
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(1, 2)
