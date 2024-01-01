
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = x1.detach()
        x3 = x2.permute(0, 2, 1)
        y2 = torch.nn.functional.linear(x3, self.linear.weight, self.linear.bias)
        x2 = torch.max(y2, dim=1)[1].unsqueeze(dim=1)
        y2 = torch.nn.functional.linear(y2, self.linear.weight, self.linear.bias)
        x2 = torch.max(y2, dim=1)[1].unsqueeze(dim=1)
        return torch.nn.functional.linear(x2, self.linear.weight, self.linear.bias)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
