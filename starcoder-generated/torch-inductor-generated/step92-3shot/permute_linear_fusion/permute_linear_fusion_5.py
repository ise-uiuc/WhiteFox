
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v1 = torch.transpose(x, 1, 2)
        v2 = torch.nn.functional.linear(v1, torch.Tensor([[-10, 200, -3], [1, 0, 2]]))
        v3 = v2.unsqueeze(1)
        v4 = torch.tile(v3, [1, 3, 1, 1])
        v5 = v4.squeeze(1)
        return v5
# Inputs to the model
x = torch.randn(1, 3, 2, 2)
