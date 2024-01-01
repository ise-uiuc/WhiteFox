
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v6 = torch.sqrt(torchvision.ops.nms(x1, x2, x3))
        return v6
# Inputs to the model
x1 = torch.randn(4, 5)
x2 = torch.Tensor(4, 5).uniform_()
x3 = torch.Tensor(4, 5).uniform_()
