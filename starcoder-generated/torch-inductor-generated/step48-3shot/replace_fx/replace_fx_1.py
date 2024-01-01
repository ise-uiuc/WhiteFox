
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y1 = torch.nn.functional.softmax(x1, dim=-1)
        y2 = torch.nn.functional.gelu(y1)
        return y2
# Inputs to the model
X = torch.randn(1, 39, 1).to(torch.float16)
