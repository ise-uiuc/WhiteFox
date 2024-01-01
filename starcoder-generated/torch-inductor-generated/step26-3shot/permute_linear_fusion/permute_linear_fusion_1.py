
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 32)
        self.weight = torch.nn.Parameter(torch.zeros(int(32)))
        self.bias = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.linear(x, self.linear.weight, self.linear.bias)
        x = torch.max(x, dim=-1)[0]
        x = x.unsqueeze(dim=-1)
        x = x + torch.max(x, dim=-1, keepdim=True)[0].to(self.weight.dtype)
        x = (x == -1).to(self.weight.dtype)
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        x = torch.sum(torch.nn.functional.hardtanh(torch.nn.functional.tanh(x), -1.0, 1.0))
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
