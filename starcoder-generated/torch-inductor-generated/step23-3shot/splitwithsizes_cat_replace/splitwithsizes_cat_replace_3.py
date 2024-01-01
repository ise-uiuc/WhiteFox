
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(3, 2, 0), torch.nn.Sequential(torch.nn.Dropout2d(p = 0.5, inplace=True), torch.nn.ConstantPad2d((1, 1, 1, 1), value=0), torch.nn.Conv2d(64, 40, 3, 1, 0)), torch.nn.MaxPool2d(3, 2, 1), torch.nn.ReLU(inplace=True), torch.nn.ConvTranspose2d(40, 64, 3, 2, 0))
        self.conv2 = torch.nn.ConvTranspose2d(64, 80, 4, 2, 1)
        self.conv3 = torch.nn.ConvTranspose2d(64, 80, 3, 1, 0)
    def forward(self, v1, v2):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (torch.cat([self.conv2(concatenated_tensor), self.conv3(v2)], dim=1), torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
x2 = torch.randn(1, 64, 64, 64)
