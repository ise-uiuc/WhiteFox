
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.nn.functional.conv2d(x1, x1, bias=None, stride=(2, 2), padding=(1, 1), dilation=(1, 1), groups=1)
        v2 = torch.nn.functional.batch_norm(v1)
        return v2 + x1
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4, 3)
