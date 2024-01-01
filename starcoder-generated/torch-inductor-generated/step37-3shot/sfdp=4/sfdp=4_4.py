
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, kernal, mask):
        qmk = x @ kernal.transpose(-2, -1)
        output = torch.softmax(qmk, dim=-1)
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 16, 16, 64)
mask = (torch.rand(1, 16, 16) > 0.7).fill_(-100000000.0)
