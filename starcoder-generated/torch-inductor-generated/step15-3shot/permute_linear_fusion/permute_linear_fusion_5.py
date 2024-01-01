
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 8, bias=False)
        self.linear2 = torch.nn.Linear(8, 8, bias=False)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1).unsqueeze(-3)
        v2 = self.linear1(v1)
        v3 = v2.permute(0, 2, 1).unsqueeze(-3)
        v4 = self.linear2(v3)
        v5 = x1 + v4.squeeze(2)
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 4)
