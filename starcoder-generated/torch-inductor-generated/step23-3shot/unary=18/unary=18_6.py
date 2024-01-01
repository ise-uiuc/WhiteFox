
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = None # Apply a pointwise convolution of kernel size 5x5
    def forward(self, x1):
        v1 = self.layer1(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 20, 50, 75)
