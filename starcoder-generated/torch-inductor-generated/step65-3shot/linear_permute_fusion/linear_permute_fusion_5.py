
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        self.linear.relu = torch.nn.ReLU(inplace=False)
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        self.tmp_v2 = v1.permute(0, 2, 1)
        self.linear.relu.inplace = True
        v2 = torch.nn.functional.linear(self.tmp_v2, self.linear.weight, self.linear.bias)
        v3 = torch.ops.aten.expand_as(v2, v1)
        self.tmp_v4 = v1 * v3
        self.linear2 = torch.nn.Linear(2, 2)
        self.linear3 = torch.nn.Linear(2, 2, device='cpu')
        self.linear.transpose2 = torch.nn.Transpose(0, 2)
        self.linear.transpose2.shape = [1, 2, 2]
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2, requires_grad=True)
