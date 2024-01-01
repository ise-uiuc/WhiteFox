
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.unsqueeze(x1, 1)
        v2 = torch.reshape(v1, (-1, 3))
        v3 = torch.matmul(v2, torch.ones(3, 1))
        v4 = torch.reshape(v3, (1,2,2))
        return torch.squeeze(v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
