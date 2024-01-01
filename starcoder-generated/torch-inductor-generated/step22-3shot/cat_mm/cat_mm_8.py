
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.mm(torch.rand(1, 5), torch.rand(5, 3))
        v2 = torch.mm(torch.rand(1, 3), torch.rand(3, 1))
        for i in range(5):
            k = torch.mm(v1, v2)
        return k
# Inputs to the model
x = torch.rand(2, 3)
