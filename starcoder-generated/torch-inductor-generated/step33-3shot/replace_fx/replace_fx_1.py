
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        x1 = torch.nn.functional.dropout(input_tensor, p=0.3)
        x2 = x1 + 1
        x3 = torch.rand_like(input_tensor, dtype=torch.float64)
        out = x2 * x3
        return out
# Inputs to the model
x1 = torch.randn(1, 2, 2)
