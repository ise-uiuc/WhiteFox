
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2, device='cuda')
        self.linear2 = torch.nn.Linear(2, 2, device='cuda')
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v2 = torch.nn.functional.relu(x1.cuda())
        v3 = x1 + v2
        v1 = torch.nn.functional.linear(v3, self.linear1.weight, self.linear1.bias)
        a1 = torch.nn.functional.linear(v3, self.linear2.weight, self.linear2.bias)
        v4 = a1.permute(0, 2, 1)
        return v4
# Inputs to the model
x1 = torch.randn(3, 2, 2, device='cuda')
