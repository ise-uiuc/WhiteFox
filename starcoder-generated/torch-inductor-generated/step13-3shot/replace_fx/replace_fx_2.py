
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.nn.functional.dropout(x1)
        x2 = torch.randn((x1.shape[0], 30))
        x3 = torch.rand((x1.shape[0]))
        x4 = x2[:, 0:23]
        return torch.mean(x4)
# Inputs to the model
x1 = torch.randn((12,12))
