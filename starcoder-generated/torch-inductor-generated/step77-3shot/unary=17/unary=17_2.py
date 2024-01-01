
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1d = torch.nn.ConvTranspose1d(
            3, 4, 15, stride=(1, 1), padding=(0,), output_padding=(0,), groups=1, bias=True
        )
    def forward(self, x1):
        v1 = self.conv1d(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 100)
