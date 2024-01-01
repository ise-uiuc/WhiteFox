
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(30, 20)
        self.avg = torch.nn.AdaptiveAvgPool2d((20, 20))
        self.conv = torch.nn.Conv2d(20, 1, 1)
        block_0 = [torch.nn.Conv2d(3, 32, 3, 2, 1), torch.nn.Conv2d(32, 32, 3, 1, 1), torch.nn.Conv2d(32, 32, 3, 3, 1)]
        block_1 = [torch.nn.Conv2d(32, 64, 3, 2, 1, bias=False), torch.nn.Conv2d(64, 64, 3, 1, 0, bias=False)]
        block_2 = [torch.nn.BatchNorm2d(64)]
        block_3 = [torch.nn.ReLU()]
        block_4 = [torch.nn.Conv2d(64, 128, (1, 1), 2, 1), torch.nn.Conv2d(128, 128, (1, 1), 1, 0)]
        block_5 = [torch.nn.BatchNorm2d(128)]
        block_6 = [torch.nn.ReLU()]
        block_7 = [torch.nn.Conv2d(128, 256, 1, 2, 1), torch.nn.Conv2d(256, 256, 1, 1, 0)]
        block_8 = [torch.nn.BatchNorm2d(256)]
        block_9 = [torch.nn.ReLU()]
        block_10 = [torch.nn.Conv2d(256, 1, (1, 1), 2, 1)]
        self.features = torch.nn.Sequential(*block_0, *block_1, *block_2, *block_3, *block_4, *block_5, *block_6, *block_7, *block_8, *block_9, *block_10)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat([split_tensors[i] for i in [0, 2]], dim=1)
        out_block0 = concatenated_tensor
        concatenated_tensor = torch.cat([split_tensors[i] for i in [0, 3]], dim=1)
        out_block1 = torch.nn.functional.relu(self.features[0](concatenated_tensor))
        concatenated_tensor = torch.cat([split_tensors[i] for i in [0, 4]], dim=1)
        out_block2 = self.features[1](concatenated_tensor)
        out_block3 = self.features[2](out_block2)
        out_block4 = self.features[3](out_block3) + out_block2
        concatenated_tensor = torch.cat([split_tensors[i] for i in [0, 5]], dim=1)
        out_block5 = self.features[4](concatenated_tensor)
        out_block6 = self.features[5](out_block5) + out_block5
        concatenated_tensor = torch.cat([split_tensors[i] for i in [0, 6]], dim=1)
        out_block7 = self.features[6](concatenated_tensor)
        out_block8 = self.features[7](out_block7) + out_block7
        concatenated_tensor = torch.cat([split_tensors[i] for i in [3, 4]], dim=1)
        out_block9 = self.features[8](concatenated_tensor)
        out_block10 = self.avg(self.conv(out_block9))
        return (out_block10,)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
