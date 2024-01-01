
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.cat((x1, v1), dim=0)
        v3 = self.linear(v2)
        v4 = v3.view([-1])
        v5 = torch.cat((v2, v3, v4), dim=0)
        v6 = v5.reshape(2, 1, 4)
        y = v6.permute(0, 2, 1)
        return y
# Inputs to the model
x1 = torch.randn(1, 2, 2)
