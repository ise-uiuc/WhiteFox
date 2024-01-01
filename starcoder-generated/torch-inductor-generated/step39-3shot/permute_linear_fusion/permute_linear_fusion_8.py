
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = v1.chunk(2, dim=1)
        v3 = [z1.squeeze(dim=1) for z1 in v2]
        x2 = torch.cat(v3, dim=0).unsqueeze(dim=-1)
        v3 = torch.max(x2, dim=-1)[0]
        v4 = v3.unsqueeze(dim=-1)
        x2 = x2 + v4.to(x2.dtype)
        v4 = (x2 == -1).to(x2.dtype)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
