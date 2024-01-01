
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module0 = torch.nn.Identity()
        self.module1 = torch.nn.ModuleList((torch.nn.Sigmoid(), torch.nn.Sigmoid()))
    def forward(self, input):
        v0 = torch.pow(input, 2)
        v1 = torch.sum(v0)
        v2 = v1 + 3
        v3 = v2.clamp(0)
        v4 = self.module0(v3)
        for v7 in self.module1:
            v5 = v4 * v3
            v6 = v7(v5)
            v4 = v6
        v8 = v4 - 0.5
        return v8
# Inputs to the model
input = torch.randn(4, 2, 3, requires_grad=True)
