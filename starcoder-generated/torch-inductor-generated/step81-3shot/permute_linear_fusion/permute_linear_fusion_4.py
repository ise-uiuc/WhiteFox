
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 5)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        x3 = x2[None, :3, 0, :, :]
        v3 = x2.detach()
        v4 = torch.max(v3, dim=-1)[1]
        v5 = v4.unsqueeze(dim=-1)
        v3 = v3 + v5.to(v3.dtype)
        v5 = (v3 == -1).to(v3.dtype)
        v3 = v3.permute(0, 2, 1)
        y3 = v5 * v3
        y4 = -1 / x3[0, :, :, :, :]
        x4 = y4[0, :, :, :, :]
        y5 = y4[0, :, :, :, :]
        x5 = y4[0, :, :, :, :]
        v6 = x5 + y5
        v7 = v6.to(v3.dtype)
        y6 = v7 + x4
        x6 = y6.squeeze(dim=0).squeeze(dim=-1)
        y7 = x2[None, :3, 2, :, :]
        x7 = y7[0, :, :, :, :]
        v8 = y3[0, :, :, :, :] * torch.cat((y6[0, :, :, :, :].unsqueeze(dim=0), x7), dim=0)
        v9 = v8[0]
        v10 = v1.permute(0, 3, 1, 2).clone()
        v11 = torch.nn.functional.pad(v10, (2, 2, 0, 0), value=float('-inf'))[None, 0, :, :, :]
        v12 = (v11 == float('-inf')).to(v11.dtype)
        v11 = torch.where(v12, torch.zeros_like(v11), v11)
        v13 = (v11!= float('-inf')).to(v11.dtype)
        v11 = torch.where(v12, v11, float('-inf'))
        v14 = (v11 < float('-inf')).to(v11.dtype)
        v15 = (v11 > float('-inf')).to(v11.dtype)
        y4 = (v9 * v14).sum(dim=-1).sum(dim=-1) + (v13 * v15).sum(dim=-1)
        v16 = y3[0, :, :, :, :] * torch.cat((x6.unsqueeze(dim=0), x7), dim=0)
        v17 = v16[0]
        y5 = (v17 * v14).sum(dim=-1).sum(dim=-1) + (v13 * v15).sum(dim=-1)
        return v16, (y5.expand_as(x2), y4.expand_as(x2))
# Inputs to the model
x1 = torch.randn(1, 3, 5, 3, 3)
