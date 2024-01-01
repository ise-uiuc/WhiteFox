
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        x2 = torch.nn.functional.relu(v2)
        v3 = torch.max(x2, dim=-1)[0]
        v4 = torch.nn.functional.softmax(torch.nn.functional.softmax(v1, dim=-1)[0].unsqueeze(dim=-1).transpose(0, 1), dim=-1)
        v5 = torch.nn.functional.relu(torch.cat([v4 for i in range(3)], dim=-1)[0])
        v5 = torch.sum(v5, dim=-1)
        v5 = torch.nn.functional.max_pool1d(torch.nn.functional.max_pool1d(v5, 3, 3), 1, 2)
        v6 = torch.sum(torch.nn.functional.pad(v5, (0, 0, 0, 0, 0, 0)))
        return torch.sum(torch.nn.functional.elu(v6.permute([0, 2, 1])))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
