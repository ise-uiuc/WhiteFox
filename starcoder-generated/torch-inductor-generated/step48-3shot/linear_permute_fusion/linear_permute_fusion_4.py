
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 4)
    def forward(self, x3):
        v3 = x3
        for i in range(2):
            if i == 0:
                v0 = v3
                v1 = torch.nn.functional.linear(v0, self.linear.weight, self.linear.bias)
                v2 = v1.permute(0, 1, 3, 2)
                v3 = v2.contiguous()
            v3.detach_()
        return v3
# Inputs to the model
x3 = torch.randn(1, 2, 2)
