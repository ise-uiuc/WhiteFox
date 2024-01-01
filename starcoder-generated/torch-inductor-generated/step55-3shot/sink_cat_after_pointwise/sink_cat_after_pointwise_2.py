
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.max(dim=1)[0].unsqueeze(dim=-1)
        y = x.mean(dim=1, keepdim=True)
        y = y.permute(1, 0, 2)
        y = x - torch.mean(y, dim=-2).unsqueeze(-2)
        return y + x.mean().unsqueeze(0)
# Inputs to the model
x = torch.randn(2, 3, 4).clone().requires_grad_()
