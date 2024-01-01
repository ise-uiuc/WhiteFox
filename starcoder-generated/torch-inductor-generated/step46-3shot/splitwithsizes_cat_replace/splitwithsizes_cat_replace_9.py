
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=2, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0, bias=False)
    def forward(self, v1):
        split_tensors = torch.split(self.conv0(v1), [1, 1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors = torch.split(concatenated_tensor, [1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors = torch.split(concatenated_tensor, [1, 1, 1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        split_tensors = torch.split(concatenated_tensor, [1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(self.conv1(v1), [1, 1, 1, 1, 1, 1, 1], dim=1), torch.split(self.conv2(v1), [1, 1, 1, 1, 1, 1], dim=1), torch.split(self.conv3(v1), [1, 1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
