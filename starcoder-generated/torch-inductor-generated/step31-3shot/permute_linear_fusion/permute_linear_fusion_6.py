
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.linear1 = torch.nn.Linear(4, 2)
    def forward(self, x1):
	v1 = x1.permute(0, 2, 1)
	v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
	v2 = torch.nn.functional.relu(v2)
	v3 = torch.arange(v2.shape[0]).to(dtype=torch.long)
	v4 = v2[v3, :]
	v5 = self.linear1(v4)
	v5 = torch.min(v5, dim=-1)[0]
	v5 = v5.unsqueeze(dim=-1)
        return v5
# Inputs to the model
x1 = torch.randn(32, 3, 4)
