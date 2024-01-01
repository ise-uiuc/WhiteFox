
class Model(nn.Module):
    def forward(self, x):
        t1 = x.view((x.shape[0], -1))
        t2 = torch.mm(t1, t1.t())
        return t2
# Inputs to the model
x = torch.randn(5, 5, 5)
