
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 1, 3, 2)
        v3 = v1.permute(0, 2, 3, 1)
        return v2, v3
# Inputs to the model
x1 = torch.randn(2, 2, 2, 2)
