
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        h1 = torch.nn.functional.gelu(x1)
        h2 = torch.nn.functional.gelu(x1)
        h3 = torch.nn.functional.gelu(x2)
        h4 = torch.nn.ReLU()(h3)
        return (h1, h4)
# Inputs to the model
x1 = torch.randn(1)
x2 = torch.randn(1)
