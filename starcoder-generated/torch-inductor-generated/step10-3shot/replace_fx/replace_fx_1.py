
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(2, 2, 2)
    def forward(self, x1):
        z1 = torch.nn.functional.dropout(self.c1(x1), p=0.5)
        z2 = torch.rand_like(self.c1(x1), dtype=torch.float)
        z1 = torch.nn.functional.dropout(torch.log_softmax(torch.nn.functional.elu(z2.abs())), p=0.3)
        return z1
# Inputs to the model
x1 = torch.randn(2, 2, 5, 5)
