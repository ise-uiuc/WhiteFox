
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.rand_like(x, dtype=torch.int64, device=torch.device("cpu:0"))
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
