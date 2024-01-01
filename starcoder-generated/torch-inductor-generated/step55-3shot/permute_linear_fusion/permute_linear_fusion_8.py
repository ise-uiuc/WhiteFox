
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight+self.linear.bias, self.linear.bias)
        v2 = torch.nn.functional.softmax(v2, dim=1)
        x2 = torch.nn.functional.log_softmax(v1, dim=1)
        v3 = x2.view(x2.shape[0], x2.shape[1])
        v4 = self.linear.weight
        v5 = torch.mean(v4, dim=0)
        v6 = torch.nn.functional.relu(v5)
        v5 = v6.permute(1, 0, 2)
        v7 = torch.mean(v5, dim=0)
        v8 = torch.nn.functional.relu(v7)
        v4 = v4.permute(0, 2, 1)
        v9 = v4 * v8
        v10 = torch.nn.functional.mean(v9, dim=1)
        v11 = torch.nn.functional.relu(v10)
        v12 = v11 * v11
        v13 = torch.nn.functional.mean(v12, dim=-1)
        v14 = torch.nn.functional.relu(v13)
        v15 = v2 + v14
        v16 = torch.clamp(v15, -2 + v6, v10)
        return v3 * v16
# Inputs to the model
x1 = torch.randn(1, 2, 2)
