
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        b1 = torch.rand((2, 2), device=x.device)
        return 2
# Inputs to the model
x = torch.randn(1)
