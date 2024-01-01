
class Model(torch.nn.Module):
    def forward(self, x, y):
        t = x.expand(x.shape[0], -1, -1)
        w = torch.bmm(t, x.transpose(1, 2))
        z = torch.bmm(x, y) + w
        t1 = torch.rand(3, 3) * z.shape[1] * z.shape[2]
        t2 = torch.rand(z.shape[1:]) * 3
        return torch.bmm(w, t1) + torch.bmm(z, t2)
# Inputs to the model
x = torch.randn(3, 4, 5)
y = torch.randn(4, 3, 6)
