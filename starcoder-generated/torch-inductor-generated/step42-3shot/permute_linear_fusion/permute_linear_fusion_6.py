
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 1)

    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.linear(v1, self.linear1.weight, self.linear1.bias)
        v4 = v2.reshape((1, 2))
        v5 = torch.nn.functional.linear(v4, self.linear2.weight, self.linear2.bias)
        v8 = v5.unsqueeze(dim=1)
        v3 = v8 + v2
        v3 = v3.contiguous()
        v3 = torch.max(v3, dim=-1)[0]
        v4 = v3.unsqueeze(dim=-1)
        v3 = v3 + v4.to(v3.dtype)
        v4 = (v3 == -1).to(v3.dtype)
        v4 = torch.nn.functional.linear(v4, self.linear3.weight, self.linear3.bias)
        v4 = torch.sum(v4)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
