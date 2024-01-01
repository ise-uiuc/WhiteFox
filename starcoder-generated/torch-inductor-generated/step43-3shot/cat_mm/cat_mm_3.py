
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x2)
        v1 = torch.cat([t1, t1], 1)
        list_t = [v1, v1, v1, v1, v1, v1]
        return torch.cat(list_t, 1)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 2)
