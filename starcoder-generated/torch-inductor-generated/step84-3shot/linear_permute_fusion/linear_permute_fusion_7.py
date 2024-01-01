
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, self.linear.weight, bias=None)
        v2 = v1.permute(0, 2, 1)
        softmax = torch.nn.Softmax(dim=-1)
        v3 = softmax(v2)
        v4 = v3.unsqueeze(2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
