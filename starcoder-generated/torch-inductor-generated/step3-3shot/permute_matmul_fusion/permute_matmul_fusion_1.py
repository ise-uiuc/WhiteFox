
class Model(torch.nn.Module):
    def __init__(self, arg):
        super().__init__()
    def forward(self, x1, input_tensor):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.bmm(v1, input_tensor)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
input_tensor = torch.randn(1, 2, 2)
