
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        t2 = torch.mul(x1, v1)
        t3 = torch.nn.functional.sigmoid(t2+x1)
        t4 = torch.nn.functional.relu(t3 + x1)
        return t4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
