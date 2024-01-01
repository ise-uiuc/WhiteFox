
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = x1.unsqueeze(1)
        v2 = torch.nn.functional.max_pool2d(v1, self.linear.kernel_size[1])
        v3 = v2.squeeze(1)
        v4 = torch.nn.functional.softmax(v3, dim=-1)
        v5 = torch.nn.functional.sigmoid(v4)
        v5 = torch.where(v5 > 0.0, torch.tensor(1.0), torch.tensor(0.0))
        v6 = x1.permute(0, 2, 1)
        return torch.nn.functional.linear(v6, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 3, 3)
