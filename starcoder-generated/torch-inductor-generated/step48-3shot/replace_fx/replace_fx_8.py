
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.gelu(x1, approximate=False)
        x3 = x2.t().contiguous()
        o1 = x3[0, :]
        o2 = F.softmax(o1, dim=0)
        o3 = F.dropout(o2, p=0.5277)
        return o3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
