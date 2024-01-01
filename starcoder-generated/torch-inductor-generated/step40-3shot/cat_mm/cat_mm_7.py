
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        v = [torch.nn.functional.adaptive_avg_pool2d(torch.randn(5, 5, 3, 3), (5, 5)), torch.nn.functional.adaptive_avg_pool2d(torch.randn(5, 5, 3, 3), (4, 4)), torch.nn.functional.adaptive_avg_pool2d(torch.randn(5, 5, 3, 3), (3, 3)), torch.nn.functional.adaptive_avg_pool2d(torch.randn(5, 5, 3, 3), (5, 5)), torch.nn.functional.adaptive_avg_pool2d(torch.randn(5, 5, 3, 3), (4, 4)), torch.nn.functional.adaptive_avg_pool2d(torch.randn(5, 5, 3, 3), (3, 3))]
        return torch.cat(v, 0)
