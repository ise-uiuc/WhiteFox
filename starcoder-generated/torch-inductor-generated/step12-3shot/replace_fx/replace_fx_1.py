
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        m = self._module
        t1 = torch.nn.functional.dropout(x, p=0.4)
        return 1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
