
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)
    def forward(self, x1):
        y = x1
        v5 = torch.nn.functional.linear(y, self.linear.weight, self.linear.bias)
        v4 = v5.permute(0, 2, 1)
        v3 = torch.nn.functional.batch_norm(v4, None, None, self.layernorm.weight, self.layernorm.bias, True, 0.019999999552965164, 1.0000000000000002e-05).contiguous()
        v1 = torch.nn.functional.layer_norm(v3, v3.size(1), self.layernorm.weight, self.layernorm.bias, 1e-05, True).contiguous()
        v2 = v1.permute(0, 2, 1)
        return v1

# Inputs to the model
x1 = torch.randn(3, 2, 2, device='cpu')
