
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, w, x1):
        m = torch.nn.functional.linear(x1, w)
        return m
# Inputs to the model
w = torch.randn(6, 6)
x1 = torch.randn(6, 6)
