
import re
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v2_channels = self._get_channel_size(v2)
        v1_channels = self._get_channel_size(v1)
        if v1_channels!= 1 and v2_channels!= 1:
            if v1_channels!= v2_channels:
                v1 = torch.reshape(v1, [1, v1_channels, v1.shape[2], v1.shape[3]])
            v1_shape = v1.shape
            v1 = torch.reshape(v1, [1, 1, v1.shape[2], v1.shape[3]])
            v1 = F.relu(v1)
            v2 = torch.reshape(v2, [1, v2_channels, v1.shape[2], v1.shape[3]])
            v2 = torch.reshape(v2, [1, 1, v2.shape[2], v2.shape[3]])
            v1 = F.relu(v1)
            v2 = F.relu(v2)
            v3 = v1 + v2
        else:
            v3 = v1 + v2
        return v3

    
    def _get_channel_size(self, x):
        if isinstance(x, torch.Tensor):
            return x.shape[1]
        else:
            try:
                return x.shape[1]
            except IndexError:
                match = re.search('Conv\((.*?),', repr(x))
                return int(match.group(1))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 16, 16)
