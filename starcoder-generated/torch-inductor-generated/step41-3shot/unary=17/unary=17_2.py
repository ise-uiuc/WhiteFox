
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 8, 3, padding=0, stride=2)
        self.conv = torch.nn.Conv2d(4, 3, (2, 4), (2, 4), 0, 1, 1)
        self.conv1 = torch.nn.Conv2d(8, 3, 7, strdiw2=2)
        self.conv2 = torch.nn.Conv2d(16, 3, 5, strdiw2=2)
        self.conv3 = torch.nn.Conv2d(32, 3, 5)
        self.conv4 = torch.nn.Conv2d(64, 3, 5, strdiw2=2)
        self.conv5 = torch.nn.Conv2d(3, 3, 9, strdiw2=2)
        self.conv6 = torch.nn.Conv2d(3, 3, 3, strdiw2=2)
    def forward(self, x0):
        v1 = self.conv_transpose(x0)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.sigmoid(v3)
        v5 = torch.relu(v2)
        v6 = v4.expand((-1, -1, 7, -1))
        v7 = v6.permute((-1, -2, -3, -4))
        v8 = v7.expand((-1, -1, 7, -1))
        v9 = self.conv2(v8)
        v10 = torch.sigmoid(v9)
        v11 = v10*v6
        v12 = v5.expand((-1, -1, 5, -1))
        v13 = v12.permute((-1, -2, -3, -4))
        v14 = v11.expand((-1, -1, 5, -1))
        v15 = self.conv3(v14)
        v16 = v5*v13
        v17 = torch.sigmoid(v15)
        v18 = v13.permute((-1, -2, -3, -4))
        v19 = v16.expand((-1, -1, 5, -1))
        v20 = v5.expand((-1, -1, 3, -1))
        v21 = v20.permute((-1, -2, -3, -4))
        v22 = self.conv4(v19)
        v23 = v21*v18
        v24 = torch.softmax(v22, dim=-1)
        v25 = v24*v16
        v26 = v25.permute((-1, -2, -3, -4))
        v27 = v13*v26
        v28 = self.conv5(v27)
        v29 = torch.sigmoid(v28)
        v30 = v21.permute((-1, -2, -3, -4))
        v31 = v25.expand((-1, -1, 3, -1))
        v32 = self.conv6(v31)
        v33 = torch.sigmoid(v32)
        v34 = v21*v30
        v35 = torch.tanh(v34)
        v36 = v21*v31
        v37 = torch.sigmoid(v36)
        v38 = torch.mul(v37, v33)
        v40 = v17*255
        return v40
# Inputs to the model
x0 = torch.randn(1, 4, 224, 224)
