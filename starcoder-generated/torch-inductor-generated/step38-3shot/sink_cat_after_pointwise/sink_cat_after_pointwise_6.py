
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(4, 2, 3)
        self.conv2d = torch.nn.Conv2d(1, 2, 3)
        self.conv3d = torch.nn.Conv3d(1, 2, 3)
    def forward(self, x):
        if (torch.randn(1) > 0.5):
            conv = self.conv1d
        elif (torch.randn(1) > 0.5):
            conv = self.conv2d
        else:
            conv = self.conv3d
        x = conv(x)
        # TODO: need to figure out input shape that makes this scenario work
        #x = x.transpose(0, 1)
        #x = x.unsqueeze(0)
        #x = x.contiguous()
        #x = x.permute(1, 2, 0)
        #x = x.transpose(2, 0)
        #x = x.squeeze()
        #x = x.permute(2, 0, 1)
        x = x.view(x.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 1, 4, 4, 4)
# Inputs to the model
y = torch.randn(4, 2, 3)
