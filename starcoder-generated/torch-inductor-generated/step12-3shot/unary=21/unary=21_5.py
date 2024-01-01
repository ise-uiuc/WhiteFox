
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 9616
        input_dim  = 3
        output_dim = 97
        stride     = 4
        padding    = 142
        self.conv = torch.nn.Conv1d(input_dim, output_dim, kernel_size, stride=stride, padding=padding )
    def forward(self, x):
        y1 = self.conv(x)
        y2 = torch.tanh(y1)
        return y2
# Inputs to the model
x = torch.randn(20, 3, 2570)
