
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 64, 16, stride=2)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(64,32,16,stride=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(32,32,16,stride=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(32,16,9,stride=1)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(16,16,9,stride=1)
        self.conv_transpose5 = torch.nn.ConvTranspose2d(16,3,9,stride=1)
    def forward(self, input):
        x = torch.relu(self.conv_transpose(input))
        y = x[:,-1]
        y = x[:,:,:]
        y = torch.clamp(y, min=0, max=255)
        return y
# Inputs to the model
input = torch.randn(1, 128, 16, 16)
