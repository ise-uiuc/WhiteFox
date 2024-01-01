
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, groups=2)
        batchnorm = nn.BatchNorm1d(2)
        relu = nn.ReLU()

        out = conv(input)
        out = batchnorm(out)
        out = out.view(out.shape[2], -1)
        out = out[:, :1]
        out = relu(out)
        return out
# Inputs to the model
input = torch.randn(1, 1, 10)
