
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, shape2):
        s1 = torch.nn.functional.dropout(input, p=0.6)
        temp = torch.rand_like(s1)
        c1 = s1.matmul(shape2)
        c2 = torch.nn.functional.dropout(temp)
        return c1
# Inputs to the model
x1 = torch.randn(4,80)
