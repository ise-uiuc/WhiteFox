
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        c = torch.nn.Conv2d(4, 4, (1, 2), 1, (0, 1), (2, 3), bias=False)
        torch.manual_seed(1)
        v2 = c(x1)
        torch.manual_seed(1)
        v1 = torch.nn.functional.batch_norm(v2, None)
        torch.manual_seed(1)
        return torch.nn.functional.batch_norm(v1, None)
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
