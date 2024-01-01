
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @torch.jit.script_method
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0.2)
        if a1.size()[1] == 2:
            return a1
        return a1 * 0
# Inputs to the model
x1 = torch.randn(1, 3, 4)
