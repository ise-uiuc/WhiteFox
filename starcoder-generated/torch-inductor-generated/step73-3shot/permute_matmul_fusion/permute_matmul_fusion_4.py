
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        t1 = x1.permute(0, 2, 1)
        t2 = x2.permute(0, 2, 1)
        t3 = torch.bmm(t1, t2)
        return t3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
