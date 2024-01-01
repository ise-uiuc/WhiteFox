
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 2).cuda()
    def forward(self, x1):
        t1 = torch.nn.functional.relu(x1)
        v1 = torch.nn.functional.max_pool2d(t1, kernel_size=1, stride=1, padding=0)
        a1 = v1.permute(0, 2, 3, 1)
        v2 = self.conv(a1)
        a2 = v2.permute(0, 3, 1, 2)
        conv1 = torch.nn.functional.conv2d
        v3 = conv1(a2, self.conv.weight, stride=2, padding=0, groups=1)
        v4 = torch.clamp(v3, min=0.0, max=10.0)
        relu1 = torch.nn.functional.relu
        v5 = relu1(v4)
        p1 = torch.nn.functional.softmax(v5, dim=1)
        v6 = p1.unsqueeze(2)
        return v6
# Inputs to the model
x1 = torch.randn(6, 3, 10, 10, device='cuda')
