
class ModelWithCustomLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d2 = torch.nn.Conv2d(2, 2, 2)
    def forward(self, x1, x2):
        x1 = torch.rand_like(x1)
        self.conv2d2(x2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 3, 2)
