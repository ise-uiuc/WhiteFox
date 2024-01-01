
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, mask, xmask):
        inp = xmask
        xmask = 0
        for i in range(inp.size(0)):
            input = inp[i]
            result = inp.new_empty((inp.size(0), 3))
            z = result[i]
            result = z
            xmask = z
        return result
# Inputs to the model
x = torch.randn((3, 2, 2), requires_grad=True)
mask = torch.randn((3) + (1,))
xmask = torch.randn(3, requires_grad=True)
