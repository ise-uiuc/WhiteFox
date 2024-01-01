
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.functional.conv_transpose2d
    def forward(self, x1):
        v1 = self.conv_transpose(input=x1, weight=torch.ones(1, 1, 4, 4), bias=None, stride=2, padding=1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 36, 36)
