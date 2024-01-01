
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear1 = torch.nn.Linear(5, 1, bias=True)
        self._linear2 = torch.nn.Linear(2, 1, bias=True)
    def forward(self, x0):
        r0 = self._linear1(x0).reshape(-1)
        r1 = torch.unsqueeze(r0, 1)
        v0 = torch.cat((r0, r1), 0)
        v1 = self._linear1(v0)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 5, 4)
