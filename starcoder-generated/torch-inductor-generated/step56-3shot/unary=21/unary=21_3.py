
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, out_channels=4, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True, padding=(0, 0))
    def forward(self, input):
        x = self.conv(input)
        x = torch.tanh(x)
        return x
# Input to the model
x = torch.ones((1, 1, 10, 10), dtype=torch.float)
