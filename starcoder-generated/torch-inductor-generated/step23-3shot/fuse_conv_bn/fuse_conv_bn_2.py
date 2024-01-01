
class Model(torch.nn.Module):
    # We are just using the PyTorch nn.Conv2d here. 
    # You can find other PyTorch nn.functional functions at https://pytorch.org/docs/stable/nn.functional.html
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 3, 1, bias=True)
    def forward(self, x):
        x = self.conv(x)
        return torch.max(x, 2)[0]
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
