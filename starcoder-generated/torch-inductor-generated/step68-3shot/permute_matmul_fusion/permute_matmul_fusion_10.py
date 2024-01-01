
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = x2.permute(0, 2, 1).unsqueeze(0)
        v4 = x1.permute(0, 2, 1).unsqueeze(0)
        v5 = torch.bmm(v1.unsqueeze(0), v2.unsqueeze(0))[0]
        v6 = torch.bmm(v3, v4)
        v7 = torch.bmm(v1[:, [0], :].unsqueeze(2), v3[:, [0], :].unsqueeze(1))
        v8 = torch.bmm(v4[:, [-1], :].unsqueeze(2), v5.unsqueeze(1))
        v9 = torch.bmm(v2[:, [0], :].unsqueeze(2), v5.unsqueeze(1))
        v10 = torch.bmm(v1.unsqueeze(0), v5.unsqueeze(0))
        v11 = torch.bmm(torch.bmm(v1.unsqueeze(0), v3.unsqueeze(0)), torch.bmm(v2.unsqueeze(0), v3.unsqueeze(0)))
        v12 = torch.bmm(v1[:, [0, 0], :], v3[:, [0, 0], :])
        v13 = torch.bmm(v2, v3)[:, [-1], :]
        v0 = torch.sum(v2 * torch.ones_like(v5))
        return (v1, v6, v7, v8, v9, v10, v11, v12, v13, v0)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
