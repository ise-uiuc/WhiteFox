
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v0 = torch.mm(x1, x1)
        v1 = torch.mm(v0, torch.mm(x2, x2))
        return v0 + v1

# Model begins
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        m0 = torch.mm(x1, x2)
        m1 = torch.mm(m0, m0)
        m2 = torch.mm(m0, m1)
        m3 = torch.mm(m1, m2)
        m4 = torch.mm(m2, m3)
        m5 = torch.mm(m3, m4)
        out = torch.mm(m4, m5)
        return out

# Model begins
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        m0 = torch.mm(x1, x1)
        m1 = torch.mm(x2, x2)
        m2 = torch.mm(m0, m1)
        m3 = torch.mm(m1, m2)
        m4 = torch.mm(m2, m3)
        m5 = torch.mm(m3, m4)
        m6 = torch.mm(m4, m5)
        m7 = torch.mm(m5, m6)
        out = torch.mm(m6, m7)
        return out
