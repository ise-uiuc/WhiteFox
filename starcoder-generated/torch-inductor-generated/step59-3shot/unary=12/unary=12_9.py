
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.sigmoid(x1)
        v2 = torch.tensor(v1, requires_grad=True)
        return x1 * v2
# Inputs to the model
x1 = torch.randn(1, 3, 57, 64)
