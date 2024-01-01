
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_tensor_A = torch.nn.Parameter(torch.randn(1, 2, 2))
        self.input_tensor_B = torch.randn(1, 2, 2)
        self.input_tensor_C = torch.randn(1, 2, 2)
    def forward(self, x1, x2, x3):
        v0 = x1.permute(0, 2, 1)
        v1 = x2.permute(0, 2, 1)
        v2 = x3.permute(0, 2, 1)
        v3 = torch.bmm(v0, v1)
        v4 = torch.bmm(v2, v1)
        return v3 + v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
x3 = torch.randn(1, 2, 2)
