
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x3):
        v0 = x3
        v1 = v0.size()
        v2 = v0.view(v1[0], -1)
        v3 = v2.permute(0, 2, 1)
        v4 = v3.contiguous()
        v5 = self.linear(v4)
        return v5
# Inputs to the model
x3 = torch.randn(1, 2, 2)
