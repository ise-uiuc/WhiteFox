
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(14, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv3 = torch.nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1) # Apply a sigmoid function to the result of the first convolution
        v3 = self.conv2(x1)
        v4 = v2 + v3 # Add the result of the first convolution with the result of the second convolution 
        v5 = self.conv3(x1)
        v6 = torch.tanh(v4 + v5) # Apply Tanh function to the result of adding the outputs from the first and third convolutions 
        return v6 
