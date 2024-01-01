
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 2, stride=1, padding=1, bias=None)
    def forward(self, x1):
        v1 = self.conv(x1)
        return torch.nn.functional.softmax(v1, dim=-1)
# Input to the model
x1 = torch.randn(1, 2, 10, 10)
