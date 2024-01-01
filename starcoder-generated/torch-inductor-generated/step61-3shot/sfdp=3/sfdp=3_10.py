
def mlp(input_tensor, hidden_size):
    v1 = torch.transpose(input_tensor, 0, -1)
    v2 = torch.nn.Linear(v1.size()[-1], hidden_size)
    v3 = v2(v1)
    v4 = torch.tanh(v3)
    v5 = torch.nn.Linear(hidden_size, v1.size()[-1])
    v6 = v5(v4)
    v7 = torch.transpose(v6, -1, 0)
    v8 = torch.nn.Linear(v7.size()[-1], hidden_size)
    v9 = v8(v6)
    v10 = torch.tanh(v9)
    v11 = torch.transpose(v10, 0, -1)
    v12 = torch.nn.Linear(v7.size()[-1], v1.size()[-1])
    v13 = v12(v11)
    v14 = torch.transpose(v13, -1, 0)
    v15 = v14 + v1
    return v15

class Model(torch.nn.Module):
    def __init__(self, hidden_size, dropout_p):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(8 * 64 * 64, 1)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.activation = torch.tanh

    def forward(self, x1, x2):
        x1_1 = self.conv(x1)
        x2_1 = self.conv(x2)
        x_mlp = mlp(torch.transpose(torch.stack((x1_1, x2_1)), 0, 1), 128)
        v1 = torch.stack((x1_1, x2_1, x_mlp))
        v1 = torch.flatten(torch.transpose(torch.cat(v1), 0, 1), start_dim=1)
        v10 = self.linear(v1)
        v2 = torch.transpose(v10, 0, -1)
        v3 = self.dropout
        v4 = self.activation
        v5 = v4(v3(v2))
        v6 = self.linear(v5)
        v7 = torch.transpose(v6, -1, 0)
        v8 = v4(v7)
        v9 = torch.transpose(v8, 0, -1)
        v10 = v6 + v9
        return v10

# Initializing the model
hidden_size = 8
dropout_p = 0.2
m = Model(hidden_size, dropout_p)

# Input tensors to the model
x1 = torch.randn(1, 6, 32, 32)
x2 = torch.randn(1, 6, 32, 32)
