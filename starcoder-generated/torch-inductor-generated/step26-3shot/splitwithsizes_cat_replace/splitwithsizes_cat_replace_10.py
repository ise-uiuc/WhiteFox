
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Sequential(torch.nn.Conv2d(59, 2, 1, stride=1), torch.nn.Softmax(dim=1))
        self.resnet = torch.nn.Sequential(torch.nn.Conv2d(2, 9, 5, 1, 4), torch.nn.Conv2d(9, 4, 5), torch.nn.Conv2d(4, 4, 5), torch.nn.Conv2d(4, 1, 1), torch.nn.Softmax(dim=1))
    def forward(self, v1):
        split_tensors = torch.split(v1, [2, 2, 2, 2, 2, 2, 2, 2, 2], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [2, 2, 2, 2, 2, 2, 2, 2, 2], dim=1))
# Inputs to the model
x1 = torch.randn(1, 59, 7, 7)
