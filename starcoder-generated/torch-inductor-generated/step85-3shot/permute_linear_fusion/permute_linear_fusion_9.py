
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.softmax = torch.nn.Softmax(-1)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v3 = v1.detach()
        v4 = v3 + 0.0030000000574829845
        v4 = torch.abs(0.9998628922826224 - v4)
        v4 = 0.0032961580163273306 * v4 * v4
        v4 = v4.mean(dim=-1)
        v4 = v4.unsqueeze(dim=1)
        v3 = v3.detach()
        v4 = v4 + v3
        v4 = torch.abs(v3 - v4)
        v4 = v4.pow(0.5)
        v5 = torch.logical_and(v4 > 0.067458583673448005, v4 < 1.1693369676343691)
        v4 = v4 * v5.float() - v5.float()
        v4 = v4.max(dim=-1, keepdim=True)[0]
        v4 = v4.pow(2)
        v4 = torch.transpose(v4, 1, 2)
        v4 = v4 + 1
        v4 = v4.matmul(v3.permute(0, 2, 1))
        v4 = v4.transpose(1, 2)
        v3 = torch.sigmoid(v3)
        v4 = 0.60000002384185791 * v4
        v3 = v4 * v3 + (1 - v4) * v1
        v3 = torch.sigmoid(v3)
        v3 = v3 * 2.081668176e-08
        v1 = v3.squeeze(dim=-1)
        return self.softmax(v1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
