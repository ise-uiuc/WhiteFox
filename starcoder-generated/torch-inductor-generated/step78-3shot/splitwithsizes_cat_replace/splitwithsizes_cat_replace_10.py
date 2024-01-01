
import torch.nn as nn
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.ModuleList([nn.Conv2d(3, 32, 3, 1, 0, bias=False), nn.ReLU(), nn.BatchNorm2d(32, affine=True, track_running_stats=True), nn.modules.pooling.AdaptiveAveragePool2d(output_size=1), nn.modules.pooling.AdaptiveMaxPool2d(output_size=(1, 1)), nn.modules.pooling.AvgPool2d(kernel_size=[1, 1], stride=1, padding=0, ceil_mode=False, count_include_pad=True), nn.modules.pooling.AvgPool2d(kernel_size=[1, 1], stride=1, padding=0, ceil_mode=False, count_include_pad=False), nn.AvgPool2d(kernel_size=[5, 5], stride=1, padding=0, ceil_mode=True, count_include_pad=True), nn.AdaptiveAvgPool2d(output_size=(1, 1))])
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
x1 = torch.randn(1, 3, 64, 64)
