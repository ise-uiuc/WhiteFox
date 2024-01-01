
class Model(torch.nn.Module):
    def forward(self, t1, t2, t3, t4):
        t5 = torch.mm(t1, t2)
        return t1 - t5 + t3 - t4
# Inputs to the model
t1 = torch.randn(6, 6)
t2 = torch.randn(6, 6)
t3 = torch.randn(6, 6)
t4 = torch.randn(6, 6)
