
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        t_var1 = torch.Tensor([1.]) # Make a temporary tensor to keep around for later use
        v3 = v1 * t_var1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
