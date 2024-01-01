
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        # import torchvision
        # import torch.autograd.profiler as profiler
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # x1 = x1.to(device)
        # model = torchvision.models.inception_v3(pretrained=True).to(device)
        # y = model(x1)
        # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        #     with profiler.record_function("model_inference"):
        #         y = model(x1)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        v1 = self.conv(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = v5 / 6
        v7 = self.bn(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
