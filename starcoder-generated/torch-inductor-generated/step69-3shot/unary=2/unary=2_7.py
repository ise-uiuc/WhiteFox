
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 5, 3, stride=2)
        self.relu = torch.nn.ReLU()
        self.maxpool2d = torch.nn.MaxPool2d(2, stride=2)
        self.conv2d = torch.nn.Conv2d(5, 6, 3, stride=1)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear = torch.nn.Linear(6*6*3, 100)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
        self.linear_1 = torch.nn.Linear(100, 10)
        self.softsign = torch.nn.Softsign()
        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.maxpool2d_1 = torch.nn.MaxPool2d(2, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 * 0.5
        v3 = v1 * v1 * v1
        v4 = v3 * 0.044715
        v5 = v1 + v4
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v7 + 1
        v9 = v2 * v8
        v10 = self.relu(v9)
        v11 = self.maxpool2d(v10)
        v12 = self.conv2d(v11)
        v13 = self.maxpool2d_1(v12)
        v14 = self.flatten(v13)
        v15 = self.linear(v14)
        v16 = self.gelu(v15)
        v17 = self.linear_1(v16)
        v18 = self.dropout(v17)
        v19 = self.softsign(v18)
        v20 = self.tanh(v19)
        v21 = self.sigmoid(v20)
        v22 = self.flatten(v21)
        v23 = self.conv2d(v22)
        v24 = self.maxpool2d(v23)
        v25 = self.conv2d(v24)
        v26 = self.maxpool2d_1(v25)
        v27 = self.flatten(v26)
        v28 = self.linear(v27)
        v29 = self.gelu(v28)
        v30 = self.linear_1(v29)
        v31 = self.dropout(v30)
        v32 = self.softsign(v31)
        v33 = self.tanh(v32)
        v34 = self.sigmoid(v33)
        v35 = self.flatten(v34)
        v36 = self.matmul(v35)
        return v36
# Inputs to the model
x1 = torch.randn(3, 2, 4, 5)
