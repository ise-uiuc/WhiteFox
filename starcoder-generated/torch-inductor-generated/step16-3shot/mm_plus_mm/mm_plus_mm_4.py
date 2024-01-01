
class Model(torch.nn.Module):
    def forward(self, t1, t2):
        tt1 = torch.mm(t1, t1)
        tt2 = torch.mm(t2, t2)
        tt3 = torch.mm(t1, t2)
        return tt1 + tt2 + tt3
# Inputs to the model
t1 = torch.randn(100, 100)
t2 = torch.randn(100, 100)
