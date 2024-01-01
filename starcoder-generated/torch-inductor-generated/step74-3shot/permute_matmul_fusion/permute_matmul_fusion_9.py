
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x1.permute(1, 0, 2)
        v1 = x2.permute(1, 0, 2)
        v2 = None
        if ((x1).view(-1)).tolist() == ((x2).view(-1)).tolist():
            v2 = v0
            if ((x1).view(-1)).tolist() == ((x2).view(-1)).tolist():
                v2 = v1
        v3 = ((v0).view(-1)).tolist() == ((v1).view(-1)).tolist()
        v6 = None
        if ((v2).permute(1, 0, 2).view(-1)).tolist() == ((v3).view(-1)).tolist():
            v6 = (torch.bmm(v2, v3.permute(1, 0, 2))).permute(1, 0, 2)
        v5 = v6
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
