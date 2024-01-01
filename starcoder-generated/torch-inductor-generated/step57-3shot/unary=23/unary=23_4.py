
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.Sigmoid()
    def forward(self):
        v1 = self.conv_transpose
        v1 = v1.conv_transpose(v1, kernel_size=3, stride=1, padding=0)
        v2 = torch.tanh(v2)
        return v2
# Inputs to the model
