
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, **kwargs):
        x3 = x1 + kwargs["x2"]
        x4 = torch.nn.functional.dropout(x3, p=0)
        return (x4, None, torch.nn.functional.dropout(x3, self._module.dropout), None, x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
