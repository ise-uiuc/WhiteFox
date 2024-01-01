
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.rand_like(x, distribution="normal")
        t2 = torch.rand_like(x, distribution="normal")
        return x + t1
# Inputs to the model
x = torch.Tensor([[0.25, 0.25, 0.25, 0.25]])
