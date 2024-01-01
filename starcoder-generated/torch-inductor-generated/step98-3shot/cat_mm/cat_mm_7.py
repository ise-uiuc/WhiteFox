
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        list_t1 = []
        for i in range(100):
            list_t1.append(v1)
        v2 = v1 * v1
        v3 = v1 + v1 + v1 + v2 - v2 - v2 - v2 # This line produces the pattern
        v1 = v1 + v1 * v1 + v1
        list_t2 = []
        for i in range(100):
            list_t2.append(v1)
        v4 = torch.cat(list_t1, 0)
        v5 = torch.cat(list_t2, 0)
        return torch.mm(v3, torch.mm(v4, v5))
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
