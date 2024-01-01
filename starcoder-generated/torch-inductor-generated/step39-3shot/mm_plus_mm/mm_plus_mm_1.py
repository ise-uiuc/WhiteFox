
class Model(torch.nn.Module):
  def forward(self, v0, v2, v3, v4, v7):
    v5 = torch.mm(v3, v7)
    v1 = torch.mm(v5, v7)
    v9 = torch.mm(v0, v7)
    v8 = torch.mm(v9, v3)
    v6 = torch.mm(v8, v5)
    v10 = torch.mm(v3, v5)
    return v2 * v6 * v1 * v10
# Inputs to the model
v0 = torch.randn(8, 8)
v2 = torch.randn(8, 8)
v3 = torch.randn(8, 8)
v4 = torch.randn(8, 8)
v7 = torch.randn(8, 8)
