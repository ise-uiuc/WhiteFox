
import collections
class Model(torch.nn.Module):
    def __init__(self, min_value=4, max_value=9):
        super().__init__()
        self.padding = collections.OrderedDict()
        self.padding['count_include_pad'] = 0
        self.padding['mode'] = 'circular'
        self.conv_transpose = torch.nn.ConvTranspose1d(1, 1, 3, stride=1, dilation=2, padding_dict=self.padding)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
