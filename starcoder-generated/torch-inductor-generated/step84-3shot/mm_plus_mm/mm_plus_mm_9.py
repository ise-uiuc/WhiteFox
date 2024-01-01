
class Model(torch.nn.Module):
    def forward(self, inp):
        t1 = torch.mm(inp, inp)
        return t1[0]
# Inputs to the model
inp = torch.randn(5, 5)
