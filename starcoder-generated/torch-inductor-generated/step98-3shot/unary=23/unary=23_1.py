
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.conv_transpose3d(input=x1, weight=torch.rand(27, 15, 3, 3, 3))
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 27, 19, 127)
