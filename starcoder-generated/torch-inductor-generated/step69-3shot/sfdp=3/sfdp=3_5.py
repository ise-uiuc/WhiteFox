
t0 = torch.randint(10, (12, 3, 4), dtype=torch.float32)
t1 = torch.randint(10, (12, 5, 4), dtype=torch.float32)
t2 = torch.randint(20, (12, 1, 6, 6, 4), dtype=torch.float32)
class Model(torch.nn.Module):
    def __init__(self, t0, t1, t2):
        super().__init__()
        self.t0 = t0
        self.t1 = t1
        self.fc0 = torch.nn.Conv3d(1, 8, 3, stride=1, padding=1)
        self.fc1 = torch.nn.Conv3d(8, 6, 3, stride=2, padding=1)
        self.fc2 = torch.nn.Conv3d(6, 12, 3, stride=1, padding=1)
        self.fc3 = torch.nn.Conv3d(12, 20, 3, stride=2, padding=1)
        self.fc4 = torch.nn.Conv3d(20, 16, 3, stride=2, padding=1)

    def forward(self, t0, t1, t2):
        v1 = self.fc0(t0)
        v2 = self.fc1(v1)
        v3 = self.fc2(v2)
        v4 = self.fc3(v3)
        v5 = self.fc4(v4)
        v6 = torch.nn.functional.max_pool3d(v5, kernel_size=(1, 4, 4), stride=(1, 4, 4))
        v7 = v6.view(-1, 1 * 4 * 4 * 4)
        v8 = torch.cat((self.t0, t1), 1)
        v9 = torch.cat((self.t1, v7), 1)
        v10 = torch.matmul(v8, v9.transpose(1, 0))
        v11 = v10 * self.t2
        v12 = v11.softmax(dim=-1)
        v13 = torch.nn.functional.dropout(v12, p=self.t2, inplace=True)
        v14 = v13.matmul(self.t2)
        return v14

# Initializing the model
m = Model(t0, t1, t2)

# Inputs to the model
x0 = torch.randn(1, 1, 4, 4)
w0 = torch.randn(3, 4)
x1 = torch.randn(1, 1, 64, 64)
w1 = torch.randn(5, 4)
__input__ = x0, w0, x1, w1
