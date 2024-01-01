
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.cat((x1, x1), dim=1)
        x3 = torch.mul(x1, x1)
        v1 = F.dropout(x2, p=0.4)
        v2 = v1.view(1, 4, 1, 1)
        v3 = torch.nn.functional.dropout(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
