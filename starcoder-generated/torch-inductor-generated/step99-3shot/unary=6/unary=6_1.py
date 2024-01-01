
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.nn.functional.avg_pool2d(x1, kernel_size=3, stride=1, padding=1)
        t2 = torch.nn.functional.conv2d(t1, num_filters=8, kernel_size=1, stride=1, padding=0)
        t3 = 3 + t2
        t4 = torch.clamp_min(t3, 0)
        t5 = torch.clamp_max(t4, 6)
        t6 = t2 * t5
        t7 = t6 / 6
        return t7.unsqueeze(dim=-1)
# Inputs to the model
x1 = torch.randn(1, 3, 541, 541)
