
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.ReflectionPad2d([0, 1, 2, 1]), torch.nn.Conv2d(6, 24, (5, 5), stride=(2, 2), padding=[0, 1, 2, 1]))
        self.avgpool1 = torch.nn.Sequential(torch.nn.ReflectionPad2d((3, 0, 3, 0)), torch.nn.AvgPool2d(2, stride=1, padding=[3, 0, 3, 0]))
        self.conv2 = torch.nn.Sequential(torch.nn.ReflectionPad2d((0, 1, 0, 1)), torch.nn.Conv2d(24, 6, (5, 5), stride=(2, 2), padding=[0, 1, 0, 1]))
    def forward(self, v1):
        split_tensors = torch.split(v1, [3, 2, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [3, 2, 1], dim=1))
# Inputs to the model
v1 = torch.Tensor(1, 6, 4, 4)
