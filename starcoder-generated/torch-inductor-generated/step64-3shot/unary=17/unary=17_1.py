
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, (5, 5), padding=(2, 2), stride=(1, 1), groups=1, output_padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.transpose(-1, -2)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.tensor([[[-1.5414, 2.7445, 4.0461, -3.1837, -1.7652]]])
