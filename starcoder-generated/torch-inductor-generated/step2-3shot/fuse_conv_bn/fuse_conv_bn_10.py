
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        norm = torch.nn.BatchNorm2d(10).eval()
        norm.num_batches_tracked = 2
        norm.running_mean = torch.ones(10).to(torch.float16)
        norm.running_mean = norm.running_mean.to(torch.float32)
        norm.running_var = (torch.ones(10) * 2).to(torch.float16)
        norm.running_var = norm.running_var.to(torch.float32)
        c = torch.nn.Conv2d(10, 10, 3).eval()
        self.layer = torch.nn.Sequential(norm, c)
    def forward(self, x):
        a = self.layer(x)
        return a
# Inputs to the model
x = torch.randn(5, 10, 100, 100, dtype=torch.float16)
