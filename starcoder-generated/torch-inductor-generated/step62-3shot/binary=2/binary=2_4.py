
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(768, 768, (1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x):
        v1 = self.conv2d(x)
        v2 = v1 - x.add(900.9009009009009)
        return v2
# Inputs to the model
x = torch.randn(1, 768, 8, 8)
