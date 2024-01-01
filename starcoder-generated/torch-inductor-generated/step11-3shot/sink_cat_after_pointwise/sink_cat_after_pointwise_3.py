
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.cat((x1, x1), dim=-1) # x1.shape: [1, 2, 2]
        v2 = torch.cat((v1, v1), dim=1)
        v3 = torch.cat((v2, v2), dim=1)
        v4 = torch.relu(v3)
        return v4.view(-1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
