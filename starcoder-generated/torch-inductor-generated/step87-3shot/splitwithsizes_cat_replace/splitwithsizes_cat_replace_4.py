
class Block(torch.nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv3d_1 = torch.nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.cat([concatenated_tensor, split_tensors[0] + concatenated_tensor], dim=1)
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = Block()
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.0), torch.nn.Dropout2d(p=0.0), torch.nn.Dropout3d(p=0.0))
        self.layer = torch.nn.Linear(1, 1, bias=True)
    def forward(self, v1):
        v2 = self.features(v1)
        v3 = torch.split(v2, [1, 1, 1], dim=1)
        v4 = torch.cat([v1, v3[1] + torch.randn(v1.shape)], dim=1)
        return v2 + self.layer(v4).relu()
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
