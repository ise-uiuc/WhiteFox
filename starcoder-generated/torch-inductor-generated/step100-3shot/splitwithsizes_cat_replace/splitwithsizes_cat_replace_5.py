
class Block(torch.nn.Module):
    def __init__(self, num_output_channels, stride):
        super(Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(128, num_output_channels, 3, stride, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return self.relu(self.conv1(concatenated_tensor))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 3, 1, padding=1)
        self.block = Block(128, 1)
        self.conv2 = torch.nn.Conv2d(128, 3, 3, 1, padding=1)
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, self.conv2(self.block(self.conv1(concatenated_tensor))))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
