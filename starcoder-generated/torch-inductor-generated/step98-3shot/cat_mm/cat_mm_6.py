
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.cat([v1], 0)
        v3 = torch.cat([v2], 1)
        list_t = []
        for i in range(100):
            list_t.append(v3)
        v4 = torch.cat(list_t, 1)
        return v4
# Inputs to the model
x1 = torch.randn(4, 5)
x2 = torch.randn(5, 6)
