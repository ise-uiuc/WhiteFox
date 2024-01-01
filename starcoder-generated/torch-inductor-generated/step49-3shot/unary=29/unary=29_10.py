
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.0, max_value=18.0):
        super().__init__()
        self.conv2d_2 = torch.nn.Conv2d(3, 1, 2, stride=1, padding=0)
        self.conv1d_1 = torch.nn.Conv1d(3, 3, 3, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.min_value
        v2 = self.conv2d_2(x)
        v3 = self.min_value
        v4 = v3 + v1
        v5 = torch.relu(v4)
        v6 = self.min_value + v1
        v7 = torch.minimum(v6, self.min_value + v2 - v1)
        v8 = v5 - v4
        v9 = self.conv1d_1(v8)
        v10 = v9.size(-4)
        v11 = v1 + torch.mean(v9, dim=-4, keepdim=True)
        v12 = v11.max(-4).values
        v13 = self.max_value - v10
        v14 = v13 - v6
        v15 = torch.min(v14, self.max_value)
        v16 = self.max_value * v15
        v17 = torch.sigmoid(v5)
        v18 = self.max_value + v12 + v17 * self.max_value
        v19 = v17.to(torch.float16)
        v20 = v18.to(torch.bfloat16)
        v21 = v20 + v5
        return v21
# Inputs as arguments
x = torch.randn(1, 3, 24)
# Input starts
input_value = x.tolist()
# Input ends
x = torch.tensor(input_value, dtype=torch.float32)
