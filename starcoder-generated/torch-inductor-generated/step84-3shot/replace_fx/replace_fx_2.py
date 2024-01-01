
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.rand_like(x, dtype=torch.float64)
        print('t1')
        x = torch.nn.functional.dropout(x, p=0.5)
    return x
# Inputs to the model
x = torch.randn(1, 2)
