
class Model(torch.nn.Module):
    def forward(self, p1, p2):
        y1 = torch.mm(p1, p2)
        z1 = torch.rand(64, 64)
        y2 = torch.mm(p1, p2)
        z2 = torch.rand(64, 64)
        y3 = y1 * y2
        z3 = z1 * z2
        y4 = torch.add(y3, y1)
        z4 = torch.add(z3, z1)
        y5 = (torch.add(y3, y4))
        z5 = (torch.add(z3, z4))
        y6 = torch.add(y5, y2)
        z6 = torch.add(z5, z2)
        y7 = torch.add(y6, torch.mm(p1, p2))
        z7 = torch.add(z6, torch.mm(p1, p2))
        y8 = torch.mm(y7, y8)
        z8 = torch.mm(z7, z8)
        y9 = 1 * 2 + 3 + 4
        z9 = 1 * 2 + 3 + 4
        y10 = x1y10z10 + x2y10z10 + x3y10z10 + 8 * 9 * 10
        z10 = x1z10z10 + x2z10z10 + x3z10z10 + 8 * 9 * 10
        y11 = x1 * y11 * y10y10z10 + x2 * z11 * z10z10
        z11 = x1 * z11 * z10z10 + x2 * y11 * y10y10z10
        q1 = 0 + 1
        f1 = q1 * 2 + x
        return y10z10y10z10 + z10z10z10y10
# Inputs to the model
p1 = 1
p2 = 1
