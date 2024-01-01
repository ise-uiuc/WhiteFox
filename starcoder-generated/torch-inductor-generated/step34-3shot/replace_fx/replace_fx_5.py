
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
# Inputs to the model
x = torch.randn(3, 2, 5, requires_grad=True)
