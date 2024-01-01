
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Identity()
        self.softmax = torch.nn.Identity()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.linear(v1)
        v3 = torch.reshape(v2, (2, 1))
        v4 = self.softmax(v3)
        v5 = v4.squeeze(dim=-1)
        v6 = v4.unsqueeze(1)
        return v1 * v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
