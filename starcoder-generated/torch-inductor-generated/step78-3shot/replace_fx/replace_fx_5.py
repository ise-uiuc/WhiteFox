
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1, dtype=torch.float, requires_grad=True)
        x3 = torch.rand_like(x3, dtype=torch.double, device='cuda')
        x4 = torch.nn.functional.dropout(x1, p=0.5, training=False)
        return x2, x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
