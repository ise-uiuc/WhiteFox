
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3), 
            torch.nn.AdaptiveMaxPool2d((3, 4))
        )
    def forward(self, x1):
        v1 = self.layers(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 64, 100, 32)
