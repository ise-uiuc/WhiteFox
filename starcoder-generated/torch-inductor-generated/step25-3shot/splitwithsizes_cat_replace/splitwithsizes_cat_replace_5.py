
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=2)
        concatenated_tensor = torch.cat((self.conv2d_2(self.relu(self.conv2d_1(split_tensors[0].squeeze(dim=2))))), dim=2)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=2))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
