
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(9, 6, kernel_size=(48, 17), stride=(1, 1), bias=False)
        self.batch_norm = torch.nn.BatchNorm2d(6)
    def forward(self, x1):
        v1 = torch.squeeze(x1, dim=0)
        v2 = self.conv_transpose(v1)
        v3 = torch.tanh(v2)
        v4 = self.batch_norm(v3)
        v5 = v4.unsqueeze(0)
        return v5
# Inputs to the model
x1 = torch.zeros(1, 1, 9, 9, dtype=torch.float32)
