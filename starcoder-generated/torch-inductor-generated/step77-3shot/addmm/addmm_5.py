
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = v1 + torch.tensor([3.], requires_grad=True)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
