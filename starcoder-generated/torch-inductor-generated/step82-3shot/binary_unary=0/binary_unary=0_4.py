
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @torch.jit.script_method
    def forward(self, x1):
        v1 = (1 + 1 + 1 + 1) + x1 + 1 + 1
        v2 = x1 + x1
        return v1
# Inputs to the model
x1 = torch.randn(4,4)
