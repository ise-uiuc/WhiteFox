
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = nn.functional.dropout(x, p=0.5, training=True)
        t2 = x
        t3 = nn.functional.dropout(t2, p=0.5, training=False)
        t4 = t3
        t5 = torch.rand(1).item()
        t6 = x
        t7 = t6 - F.dropout(t6, p=t5, training=True)
        t8 = t7
        t9 = torch.rand(1).item()
        t10 = F.dropout(t8, p=t9, training=False)
        x = torch.mean(torch.cat((t3.reshape([1, -1]), t10.reshape([1, -1])), dim=1))
        return torch.cos(x), t4
# Inputs to the model
x1 = torch.randn(1, 1)
