
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, mask):
        xz = x @ y.transpose(-2, -1)
        qk = xz / torch.sqrt(xz.size(-1)) 
        qk = qk + mask
        softmax = torch.softmax(qk, dim=-2)
        z = softmax @ x
        return z
# Inputs to the model - Inputs to the model
Input = torch.randn(1, 16, 64)
Weight = torch.randn(1, 16, 128)
Mask = torch.randn(1, 16, 64) == 0
