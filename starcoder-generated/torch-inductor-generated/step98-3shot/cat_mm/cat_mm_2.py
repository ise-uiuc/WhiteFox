
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        list_t = []
        for i in range(100):
            list_t.append(v1)
        v3 = torch.mm(v2, x2)
        return torch.cat(list_t, 1)
# Inputs to the model
x1 = torch.randn(10, 2)
x2 = torch.randn(10, 2)
