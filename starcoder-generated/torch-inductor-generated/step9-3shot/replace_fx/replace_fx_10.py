
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.4)
        x3 = x1.clamp(min=-1.0)
        x4 = torch.nn.functional.dropout(x1, p=0.3)
        x5 = torch.nn.functional.dropout(x3, p=0.8)
        x6 = x3 + x2
        return x6
# Inputs to the model
x1 = torch.randn(-1, -2)
