
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        for i in range(10):
            for j in range(3):
                v = torch.bmm(x1, x2)
        return torch.squeeze(v)
# Inputs to the model
x1 = torch.randn(2, 2, 2)
x2 = torch.randn(2, 2, 2)
