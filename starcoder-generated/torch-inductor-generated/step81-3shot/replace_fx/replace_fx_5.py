
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.nn.functional.dropout(x1, p=0.5)
        t2 = torch.rand_like(t1) # Use random numbers here
        t3 = torch.rand_like(t1)
        t4 = torch.rand_like(t1)
        t5 = torch.rand_like(t1)
        x2 = 0.0 * t1 + t2 + t3 + t4 + t5
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 3)
