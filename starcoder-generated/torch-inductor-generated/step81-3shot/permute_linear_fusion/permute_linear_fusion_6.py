
class T(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.randn(1, 119, 119, device="cuda")
        return v1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.permute = torch.nn.Sequential(T())
    def forward(self, x1):

        w4 = torch.randn(1, 119, 119, device="cuda")
 
        v1 = self.permute(x1)


        v2 = w4.permute(0, 2, 1)

        v3 = w4.unsqueeze(1)

        v3 = v3.expand(-1, 119, -1, -1)

        v4 = (w4 < 0)

        v5 = (w4 < 0).long()

        v5 = v5.to(w4.dtype)

        v5 = v5.unsqueeze(dim=-1)

        v5 = v5.unsqueeze(dim=1)

        v5 = v5.expand(-1, 119, 1, 2)

        v5 = (v5 * (-1))

        v4 = v4.to(w4.dtype)

        v4 = v5 * v4

        v4 = v4.sum(-1, keepdim=False)

        v4 = v4.unsqueeze(-1)

        v5 = v4.abs()

        v5 = (-1) * v5

        v6 = v4 * v5

        v6 = v6.unsqueeze(-1)

        v7 = v3 + v6

        v3 = v7.permute(0, 2, 1, 3)

        v8 = v3 + v2

        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 119, 119, device="cuda")
