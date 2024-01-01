
from torch.utils.data import DataLoader
import torch
class Model(torch.nn.Module):
    def __init__(self, input_size, num_channels, maxpool_stride, num_classes):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose3d(in_channels=input_size, out_channels=num_channels, kernel_size=(8, 9, 4), stride=(2, 2, 2))
        self.conv_t2 = torch.nn.ConvTranspose3d(in_channels=num_channels, out_channels=num_channels, kernel_size=(8, 10, 6), stride=(1, 2, 2))
        self.avgpool = torch.nn.AvgPool3d(kernel_size=maxpool_stride, stride=maxpool_stride)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(num_channels, num_classes)
    def forward(self, x):
        x1 = self.conv_t1(x)
        x2 = torch.relu(x1)
        x3 = self.conv_t2(x2)
        x4 = torch.relu(x3)
        x5 = self.avgpool(x4)
        x6 = self.flatten(x5)
        x7 = self.fc(x6)
        return x7
input_size = 512
num_channels = 56
maxpool_stride = 2
num_classes = 5
# Inputs to the model (set of 16 examples)
input_data = torch.randn(16, input_size, 32, 48, 64)
# Output of the model (16 examples)
output_data = torch.randn(16, num_classes)
dataset = torch.utils.data.TensorDataset(input_data, output_data)
dataloader = DataLoader(dataset, batch_size=64)
model = Model(input_size, num_channels, maxpool_stride, num_classes)
