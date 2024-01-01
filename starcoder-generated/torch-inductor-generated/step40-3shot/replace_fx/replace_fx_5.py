
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        x1 = 1 - input_tensor
        x2 = x1 * 2
        x3 = torch.cat((x2, x1), dim=0)
        x4 = torch.nn.functional.dropout(x3)
        x5 = torch.sum(x4)
        out = x5 * 3
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
