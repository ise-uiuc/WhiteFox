
def func(input):
    size = [1, 5, 5]
    x = torch.rand(1, 3, 10, 10)
    t1 = torch.addmm(input, x, x) / 4
    t2 = torch.cat([t1], dim=1)
    return t2
# Inputs to the model
input = torch.randn(3, 3)
