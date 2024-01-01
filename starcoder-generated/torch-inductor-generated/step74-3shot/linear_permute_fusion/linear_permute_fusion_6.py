
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.relu = torch.nn.ReLU6()
        self.linear_2 = torch.nn.Linear(6, 1).cuda()
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.squeeze(v1)
        v3 = self.relu(v2)
        v4 = torch.sigmoid(v3)
        a1 = torch.tanh(v3)
        v5 = torch.matmul(v3, v3)
        v6 = self.relu(v1)
        v7 = v5.permute(2, 1, 0)
        b1 = torch.where(v1 > 0, v3, v2)
        v8 = torch.log(v2)
        v9 = v6.permute(0, 2, 1)
        v10 = v8.permute(1, 0)
        v11 = self.linear_2(a1)
        return v11
# Inputs to the model
x1 = torch.randn(1, 1, 2, device='cuda')
