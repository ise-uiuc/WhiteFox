
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        x1 = torch.nn.functional.dropout(input_tensor, p=0.3)
        x2 = x1 + 1
        x3 = torch.nn.functional.dropout(x2, p=0.3)
        x4 = x3 + 1
        x5 = torch.nn.functional.dropout(x4, p=0.3)
        x6 = x5 + 1
        x7 = torch.rand_like(input_tensor)
        out = x7 / (x6 - x5)
        return out
# Inputs to the model
x1 = torch.randn(1, 2, 2)
