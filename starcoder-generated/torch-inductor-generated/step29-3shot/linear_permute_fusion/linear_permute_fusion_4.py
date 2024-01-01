
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(17, 2048)
        self.linear2 = torch.nn.Linear(2048, 2048, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear1.weight)
        v2 = v1.permute(0, 2, 1)
        v3 = torch.tanh(v2)
        v4 = v3.permute(0, 2, 1).contiguous()
        v5 = torch.tanh(v4)
        v6 = v5.permute(2, 0, 1)
        v7 = torch.nn.functional.linear(v6, self.linear2.weight)
        return v7
# Inputs to the model
x1 = torch.randn((1 << 10) + 1, 17, 1)
