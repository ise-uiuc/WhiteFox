
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        return torch.rand_like(x1) * x2.unsqueeze(1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(1,)
