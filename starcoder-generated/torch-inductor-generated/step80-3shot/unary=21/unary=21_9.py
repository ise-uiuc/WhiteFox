
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        r1 = torch.randperm(x.nelement())
        a = x.new(r1.shape).select(-1, 0)
        b = x.new(r1.shape).select(-1, 1)
        c = x.new(r1)
        c = torch.remainder(c.reshape(x.nelement()), 3)
        d = x.new(c.shape).select(-1, 2)
        e = x.new(torch.sum(c!= d))
        a = r1.select(-1, d-d*e)
        c = a.reshape(x.nelement())
        f = x.new(torch.sum(a!= b))
        g = torch.randint(high=256, size=(f,))
        h = torch.randint(high=256, size=(f,))
        c = torch.gather(a, 0, c)
        c = torch.gather(h, 0, c)
        c = c.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        i = x.new_zeros(())
        i = torch.ops.aten.native_dropout(i, p=0.5, train=True)
        x = x*i
        b = torch.randn_like(c)
        x = torch.add(x.sin(), b)
        x = torch.ops.aten.mean(x, 1)
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(4096, 4096, 1, 1, device='cuda')
