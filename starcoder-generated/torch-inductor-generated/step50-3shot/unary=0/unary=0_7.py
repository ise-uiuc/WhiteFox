
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1_weight_bias = torch.nn.Parameter(torch.randn(1, 882, 787, 82))
    def forward(self, x7):
        v1 = self.conv1x1_weight_bias
        v2 = x7 * v1
        return v2
# Inputs to the model
x7 = torch.randn(1, 82, 628, 118)
