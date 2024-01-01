
class Model(torch.nn.Module):
    def forward(self, x):
        t1 = torch.randint(0, 10, size=[1])
        t2 = torch.nn.functional.dropout(x, 0.2, False)
        t3 = torch.rand_like(t2)
        t3 = torch.nn.functional.dropout(x, t2[0], True)
        t5 = t1 + t2[0]
        return x * t2
# Inputs to the model
x = torch.randn((2, 2))
