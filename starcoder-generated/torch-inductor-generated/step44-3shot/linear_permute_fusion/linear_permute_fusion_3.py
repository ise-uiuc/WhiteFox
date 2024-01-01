
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0):
        v0 = torch.nn.functional.linear(x0, torch.tensor([[1.]], device="cuda", dtype=torch.float16), bias=None)
        v1 = v0.permute(0, 2, 1)
        v2 = v1.bool()
        return v2
# Inputs to the model
x0 = torch.randn(2, 2, 2, dtype=torch.float32)
