
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(20, 80)
        self.conv = torch.nn.Conv2d(80, 256, kernel_size=(1, 3), stride=(1, 3), padding=(0, 2), dilation=(1, 1), groups=1)

    def forward(self, x):
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        x = x.view(x.size(0), x.size(2), x.size(1))
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(x.size(0), x.size(2), x.size(1))
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 20, 10)
