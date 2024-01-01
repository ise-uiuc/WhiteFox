
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = (torch.nn.functional.avg_pool2d)(x1, [9])
        v2 = torch.view_as_real(v1)
        v3 = v2 - torch.randn(4, 4)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)
