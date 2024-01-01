
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1)
    def forward(self, input):
        output = self.conv2(input)
        return output[:,0,:,:], output[:,1,:,:], output[:,2,:,:]
# Inputs to the model
input = torch.randn(1, 3, 224, 224)
