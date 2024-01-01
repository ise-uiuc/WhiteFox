
class Model(nn.Module):
    def forward(self, inp):
        t0 = torch.mm(inp, inp)
        t1 = torch.mm(inp, inp)
        t2 = torch.mm(inp, inp)
        return (
            t0
            + t1
            + t2
            + torch.mm((t0 + t1 + t2), inp)
            + torch.mm(inp, (t0 + t1 + t2))
        )
# Inputs to the model
inp = torch.randn(200, 200)
