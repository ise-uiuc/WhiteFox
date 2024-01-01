
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v1 = v1.permute(0, 2, 1)
        v1 = v1.permute(0, 2, 1)
        # Please replace the next line with a proper use of torch.matmul
        v2 = torch.add(x1, v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
