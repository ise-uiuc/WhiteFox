
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=1)
    def forward(self, *input):
        tensor0 = input[0]
        tensor1 = self.conv(tensor0)
        tensor2 = tensor1 + 3
        tensor3 = torch.clamp(tensor2, 0, 6)
        tensor4 = tensor1 * tensor3
        v6 = tensor4 / 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
