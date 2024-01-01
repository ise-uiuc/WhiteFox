
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        for _ in range(2):
            t11 = torch.nn.functional.dropout(x, p=0.8)
            t12 = torch.rand_like(x)
# Inputs to the model
x = torch.randn(1, 2)
