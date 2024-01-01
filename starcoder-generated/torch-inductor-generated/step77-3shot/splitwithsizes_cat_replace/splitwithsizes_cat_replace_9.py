
class Block1(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.upsample = torch.nn.Upsample(size=(1,2), mode='nearest')
        self.conv1 = torch.nn.Conv2d(channel_size, int(2*channel_size), 1, 1, 0, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        v2 = self.upsample(concatenated_tensor)
        return self.conv1(v2)
class Block2(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.concat = torch.nn.ModuleList([
            torch.nn.Conv2d(int(2*channel_size), 2*channel_size, 1, 1, 0, bias=False),
            torch.nn.Conv2d(int(2*channel_size), 2*channel_size, 1, 1, 0, bias=False),
            torch.nn.Conv2d(int(2*channel_size), 2*channel_size, 1, 1, 0, bias=False),
        ])
    def forward(self, v1, v2, v3):
        v4 = torch.cat([v1, v2, v3], dim=1)
        return self.concat[i](v4)
class Block3(torch.nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channel_size, channel_size, 3, 1, 1, bias=False)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channel_size, channel_size, 3, 1, 1, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(channel_size, channel_size, 3, 1, 1, bias=False)
        self.relu2 = torch.nn.ReLU()
    def forward(self, v2, v3):
        return self.conv3(self.relu2(self.conv2(self.relu1(self.conv1(v3)))))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features_0 = [torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False), torch.nn.ReLU()]
        self.features = torch.nn.Sequential(*self.features_0)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors0, split_tensors1 = torch.split(split_tensors, [1], dim=0)
        concat_tensor = torch.nn.ModuleList([x for xs in split_tensors for x in xs])
        v1 = v1
        v1 = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)(v1)
        v2 = self.features(concatenated_tensor)
        v3 = (None, v2)
        v3 = Block1(32)(concat_tensor[0])
        v3 = Block2(32)(v1, concat_tensor[1], v3)
        v3 = Block3(32)(v1, v3)
        v3 = Block2(32)(v1, concat_tensor[2], v3)
        v3 = Block3(32)(v1, v3)
        return (concatenated_tensor, v3)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
