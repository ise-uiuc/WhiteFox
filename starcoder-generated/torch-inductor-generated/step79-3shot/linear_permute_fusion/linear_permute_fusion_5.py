
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.clone()
        v1 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 1, 2)
        v3 = torch.nn.functional.linear(v2[:,0], self.linear.weight, self.linear.bias)
        v4 = v2.transpose(1, 2)
        v4 = torch.nn.functional.linear(v4[:,:,1], self.linear.weight, self.linear.bias)
        v5 = v1 + v3
        return v4 + v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
