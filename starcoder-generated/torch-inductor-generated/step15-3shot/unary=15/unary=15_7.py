
# PyTorch 1.7 introduces a new parameter named groups in a convolutional layer.
# Please uncomment and generate the new models accordingly.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 2, 1, stride=1, padding=0, groups=2)
    def forward(self, x1):
        v1 = torch.relu(self.conv(x1))
        return v1
# Inputs to the model
x1 = torch.randn(1, 6, 32, 32)
