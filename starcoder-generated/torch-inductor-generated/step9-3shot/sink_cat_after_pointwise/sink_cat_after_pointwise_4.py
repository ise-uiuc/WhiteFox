
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat((x1, x1), dim=1)
        v2 = torch.cat((v1, v1), dim=1)
        v3 = v2.tanh() if torch.numel(v1) == 2 else v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 2)
