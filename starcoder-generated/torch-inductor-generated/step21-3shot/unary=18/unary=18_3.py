
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d([1,3,6], 3, [3,5,], 1, padding=[1,2])

    def forward(self, x1):
        v1 = self.conv(x1)
        return v1

# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
