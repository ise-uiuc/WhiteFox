
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1[:, :, 0, 0] # Select the output on the first element of the tensor
        v3 = v2[0:size] # Slice along dimension 0
        v4 = torch.cat([v1, v3], dim=1) # Concatenate the original convolutional output and the sliced output
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
