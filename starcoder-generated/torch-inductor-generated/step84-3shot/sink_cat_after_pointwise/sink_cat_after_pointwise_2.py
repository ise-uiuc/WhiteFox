 (This model does not trigger a fusible pattern)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        y = torch.cat(tensors=[x1, x2], dim=1)
        z = y - y
        return z
# Inputs to the model
x1 = torch.randn(2, 3, requires_grad=True)
x2 = torch.randn(2, 3, requires_grad=True)
