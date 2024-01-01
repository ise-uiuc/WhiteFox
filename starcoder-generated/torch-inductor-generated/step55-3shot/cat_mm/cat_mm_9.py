
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = []
        v1.append(torch.mm(x1, x1))  + [torch.mm(x1, x1)]
        return torch.cat(v1 * 3, 1)
# Input to the model
x1 = torch.randn(5, 5)
