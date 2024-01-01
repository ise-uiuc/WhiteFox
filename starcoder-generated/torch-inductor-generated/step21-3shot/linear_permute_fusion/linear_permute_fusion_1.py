
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
    def forward(self, x1) -> Any:
        out = torch.zeros_like(x1, device = self.linear.weight.device)
        out[:, 0] = self.linear(x1[:, 1])
        out[:, 1] = 1
        out[:, 2] = self.linear(x1[:, 2])
        out = out.permute(2, 0, 1)
        return out
# Inputs to the model
x1 = torch.randn(3, 3, 2).cuda()
