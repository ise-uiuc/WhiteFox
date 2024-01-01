
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.bmm(x1.permute(0, 2, 1), x2.permute(0, 2, 1))
        return v1
# Inputs to the model (batch_size, channel, spacial dimensions)
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
