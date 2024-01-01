
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x11 = x1.detach().permute(0, 2, 1)
        x22 = x2.detach().permute(0, 2, 1)
        return torch.stack((x11, x22))
# Inputs to the model
x1 = torch.randn(1, 2, 2, requires_grad=True)
x2 = torch.randn(1, 2, 2, requires_grad=True)
