
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.functional.conv_transpose2d
    def forward(self, x1):
        v1 = self.conv_t(x1, weight=torch.tensor([[[[-0.7576]], [[-0.8314]], [[-0.7894]]], [[[-0.3408]], [[-0.6156]], [[-0.3063]]], [[[1.2174]], [[1.1322]], [[1.2979]]]]), bias=None, stride=(1, 94), padding=(53, 52), dilation=(1, 93))
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(79, 1, 18, 57)
