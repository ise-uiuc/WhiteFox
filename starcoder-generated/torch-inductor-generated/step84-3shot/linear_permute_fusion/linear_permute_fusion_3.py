
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = torch.nn.functional.relu(self.linear.forward(x1))
        v2 = v1.permute(0, 2, 1)
        softmax1 = torch.nn.functional.softmax
        v3 = softmax1(v2, dim=-1)
        v4 = v3.permute(0, 2, 1)
        softmax2 = torch.nn.functional.softmax
        v5 = softmax2(v4, dim=-1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
