
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2048, 256)
        self.linear1 = torch.nn.Linear(256, 512)
    def forward(self, x1):
        v7 = torch.nn.functional.linear(x1, self.linear.weight * 0.13, self.linear.bias * 0.27)
        v9 = torch.nn.functional.linear(v7, self.linear1.weight * 0.11, self.linear1.bias * 0.37)
        v8 = torch.nn.functional.relu(v9)
        v3 = torch.max(x1, dim=-1)[0]
        v4 = v3.unsqueeze(dim=-1)
        v1 = v3 + v4.to(v3.dtype)
        v4 = (v1 == -1).to(v3.dtype)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2048, 2)
