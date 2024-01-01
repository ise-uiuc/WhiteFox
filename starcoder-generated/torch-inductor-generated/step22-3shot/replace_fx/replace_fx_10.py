
class model(torch.nn.Module):
    def __init__(self):
         self.batch_norm = torch.nn.BatchNorm2d(4)
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.max_pool2d(self.batch_norm(x1), kernel_size=2, stride=2, padding=0, ceil_mode=False)
        x3 = torch.rand_like(x1)
        return x3
# Inputs to the model
x1 = torch.randn(3, 4, 4)
