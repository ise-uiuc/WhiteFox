
class Model(torch.nn.Module):
    def forward(self, t):
        p = torch.mm(t, t)
        u = p + t
        v = torch.mm(u, t)
        w = v + u
        x = torch.mm(w, w)
        y = x + 1
        return y # Here t is returned, however the model does not return any specific part
# Inputs to the model
t = torch.randn(100, 100)
