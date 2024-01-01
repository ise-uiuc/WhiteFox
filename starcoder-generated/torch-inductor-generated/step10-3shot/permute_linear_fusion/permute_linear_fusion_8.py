
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
    def forward(self, input_tensor: torch.Tensor, **kwargs):
        v2 = input_tensor
        v1 = v2.permute(0, 2, 1)
        v3 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v3
# Inputs to the model
input_tensor = torch.randn(1, 2, 5)
