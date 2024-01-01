
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 10)
    def forward(self, x1):
        v1 = _torch_operator_add_1(x1, self.linear.bias.unsqueeze(0).expand(x1.size(0), 10))
        v2 = _torch_operator_add_2(x1, 3)
        v3 = _torch_operator_add_3(x1, v2)
        v4 = _torch_operator_add_4(v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 4)
