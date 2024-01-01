
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        return x1 + torch.randn_like(x1) @ torch.randn(5, 5)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
