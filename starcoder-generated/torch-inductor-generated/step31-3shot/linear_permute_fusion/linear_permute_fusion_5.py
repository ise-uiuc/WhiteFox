
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
    def forward(self, x1):
        v1 = torch.cat([torch.stack([torch.zeros_like(x1) for i in range(torch.numel(x1))], dim=0) for i in range(torch.numel(x1))])
        v1 = torch.stack([v1 for i in range(torch.numel(x1))])
        v1 = torch.rand(2 * torch.numel(x1), device='cpu')
        v1 = v1 - v1
        v2 = v1.view(2, 2, torch.numel(x1), 1)
        v2 = torch.stack([v2.permute(1, 0, 2, 3) for i in range(2)])
        v3 = torch.stack([v2.view(4, torch.numel(x1), 1) for i in range(2)])
        v4 = v3.permute(2, 1, 0)
        v4 = v4.permute(0, 2, 1)
        return v4
# Inputs to the model
x1 = torch.randn(2, 2, 2, device='cpu')
