
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(16, 33, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(33, 65, 3, stride=2)
        self.fc = nn.Linear(2304, 10)
    def forward(self, x):
        x = self.conv(x)
        x = self.conv2(x)
        # flatten the tensor
        x = x.view(-1, 65 * 13 * 13)
        x = torch.mm(x, x.t())
        x /= x.max()
        x = torch.mm(x, x.t())
        x /= x.max()
        x = torch.mm(x, x.t())
        x /= x.max()
        o = self.fc(x)
        return torch.mm(o, o.t())
# Inputs to the model
x = torch.rand(1, 16, 28, 28) # The size of the 4d tensor is [batch_size, num_input_channels, H, W]
