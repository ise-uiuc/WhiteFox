
class Model(torch.nn.Module):
    def __init__(self, input_channels=3, output_channels=2, kernel_size=3, padding=0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=padding)
        self.max_pool = torch.nn.MaxPool2d(2, stride=2, padding=0) 

    def forward(self, x):
        __split_tensors = torch.split(torch.relu(self.conv1(x)), output_channels, dim=1)
        __concatenated_tensor = torch.cat(__split_tensors, dim=1)
        return torch.flatten(__concatenated_tensor, 1)


# Initializing the model
m = Model(output_channels=128)

# Inputs to the model
x = torch.randn(1, 3, 224, 224)


