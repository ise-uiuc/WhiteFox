
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 9, (3, 4), dilation=(3, 2))
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout2d(p=0.2, inplace=True)
        self.batch_norm = torch.nn.BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.relu(v1)
        v3 = self.dropout(v2)
        v4 = v3.reshape(1, 1, 3 * -2, 4 * -2)
        v5 = self.batch_norm(v4)
        v6 = v5[:, :, :v5.shape[2] - 1, :v5.shape[3] - 1]
        v7 = self.max_pool(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 50, 57)
