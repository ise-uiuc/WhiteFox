
class TransposeViewConcat(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.transpose(2, 3)
        y = torch.cat([x, x, x], dim=0)
        return y
# Inputs to the model
x = torch.randn(2, 3, 5, 7)
