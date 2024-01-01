
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 32, 4, stride=0, padding=0, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(32, 64, 4, stride=4, padding=1, output_padding=0)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose5 = torch.nn.ConvTranspose2d(128, 128, 1, stride=1, padding=0, output_padding=0)
        self.conv_transpose6 = torch.nn.ConvTranspose2d(128, 1, 2, stride=1, padding=0, output_padding=0)

    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv_transpose3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv_transpose4(v6)
        v8 = torch.tanh(v7)
        v9 = self.conv_transpose5(v8)
        v10 = torch.softmax(v9, dim=1)
        v11 = self.conv_transpose6(v10)
        v12 = torch.sigmoid(v11)
        v13 = v12.view(-1)
        return v13
# Inputs to the model
import numpy as np
x1 = torch.from_numpy(np.arange(0, 19552).reshape(1,32,32).astype(np.float32))
