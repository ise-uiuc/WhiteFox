
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = torch.nn.AdaptiveMaxPool2d(output_size=(1, 4))
    def forward(self, x1):
        v1 = self.max_pool(x1)
        v2 = v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 4, 64)
