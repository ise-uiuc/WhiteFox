
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight.to(torch.float16), self.linear.bias.to(torch.float16))
        v2 = v1.permute(0, 1, 3, 2).to(torch.float16)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
