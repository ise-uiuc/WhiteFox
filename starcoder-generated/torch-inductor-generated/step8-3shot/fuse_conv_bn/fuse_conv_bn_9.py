
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m0 = torch.nn.Conv2d(2, 2, 3)
        torch.manual_seed(3)
        m0.weight = torch.nn.Parameter(torch.randn(m0.weight.shape))
        m0.bias = torch.nn.Parameter(torch.randn(m0.bias.shape))
        self.m1 = torch.nn.Sequential(m0)
    def forward(self, x):
        v1 = self.m1(x)
        y2 = torch.nn.functional.batch_norm(v1, torch.full_like(v1, 3.0, dtype=torch.float), torch.full_like(v1, 3.0, dtype=torch.float), torch.full_like(v1, 3.0, dtype=torch.float), torch.full_like(v1, 3.0, dtype=torch.float), torch.full_like(v1, 3.0, dtype=torch.float), False, 0.1, torch.backends.cudnn.benchmark)
        y3 = torch.nn.functional.batch_norm(v1, torch.full_like(v1, 3.0, dtype=torch.float), torch.full_like(v1, 3.0, dtype=torch.float), torch.full_like(v1, 3.0, dtype=torch.float), torch.full_like(v1, 3.0, dtype=torch.float), torch.full_like(v1, 3.0, dtype=torch.float), False, 0.1, torch.backends.cudnn.benchmark)
        return v1
# Inputs to the model
x = torch.randn(2, 2, 5, 6)
