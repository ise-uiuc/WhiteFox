
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(125, 20, kernel_size=(1, 10), padding=(0, 0))
        self.conv1_weight = torch.nn.Parameter(torch.empty(20, 125, 120))
        self.conv1_bias = torch.nn.Parameter(torch.empty(20))
        self.conv1 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.conv2 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.conv3 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.conv4 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.conv5 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.conv6 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.conv7 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.conv8 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.avg_pool = torch.nn.AvgPool1d(1, stride=2, padding=0)
        self.conv9 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.conv10 = torch.nn.Conv1d(20, 20, kernel_size=(1, 1), padding=(0, 0))
        self.flatten = torch.nn.Flatten()
    def forward(self, x1):
        x1 = x1.permute(0, 2, 1)
        # self.conv.reset_parameters()
        out_channels = self.conv.out_channels
        in_channels = self.conv.in_channels
        if in_channels!= x1.shape[1]:
            self.conv.in_channels = x1.shape[1]
            self.conv._create_weight_and_bias(in_channels)
        if out_channels!= x1.shape[1]:
            self.conv.out_channels = x1.shape[1]
            self.conv._create_weight_and_bias(x1.shape[1])
        x = self.conv(x1)
        out = F.relu(x)
        if out.numel() == 0:
            return out
        x = x.permute(0, 2, 1)
        z2 = out.contiguous().view(out.numel())
        v1 = torch.nn.functional.linear(z2, self.conv1_weight, self.conv1_bias)
        v1 = F.hardsigmoid(v1)
        if v1.numel() == 0:
            return v1
        out = self.avg_pool(out)
        out = out + v1
        v2 = out.permute(0, 2, 1)
        out = torch.nn.functional.hardsigmoid(v2)
        out = out.permute(0, 2, 1)
        return F.hardsigmoid(out)
# Inputs to the model
x1 = torch.randn(1, 125, 120)
