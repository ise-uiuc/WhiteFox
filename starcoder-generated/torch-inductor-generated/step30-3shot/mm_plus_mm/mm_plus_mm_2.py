
class Model(torch.nn.Module):
    def forward(self, t):
        t1 = t + t + t + t
        t2 = torch.mm(torch.mm(t, t), t)
        t3 = torch.mm(t, t)
        return t1.mm(t2) + t3
# Inputs to the model
t = torch.randn(10, 10)
