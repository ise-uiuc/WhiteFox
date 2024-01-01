
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=True)
    def forward(self, x1):
        v1 = self.pool(x1) # Change `pool` to either `MaxPool2d` or `AvgPool2d`
        return v1
# Inputs to the model
x1 = torch.randn(1, 8, 85, 85)
