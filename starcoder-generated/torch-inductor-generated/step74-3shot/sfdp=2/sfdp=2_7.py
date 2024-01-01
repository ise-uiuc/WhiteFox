
class Model(torch.nn.Module):
    def __init__(self, q_chw, k_chw, v_chw):
        super().__init__()
        q_dim, q_channels, q_height, q_width = q_chw
        k_dim, k_channels, k_height, k_width = k_chw
        v_dim, v_dim, v_height, v_height = v_chw
        assert q_dim == 0
        assert q_channels == v_channels

        self.conv_q = torch.nn.Conv1d(q_channels, k_channels, 1, stride=1, padding=1)
        self.conv_k = torch.nn.Conv1d(k_channels, k_channels, k_height, stride=1, padding=1)
        self.conv_v = torch.nn.Conv1d(k_channels, v_channels, k_height, stride=1, padding=1)

        self.conv_o = torch.nn.Conv1d(v_channels, v_channels, 1, stride=1, padding=1)

    def forward(self, x1, x2):
        v1 = self.conv_q(x1)
        v2 = self.conv_k(x2)
        v3 = self.conv_v(x2)
        v4 = torch.matmul(v1, v2.transpose(-2, -1)) 
        v5 = v4.div(10)
        v6 = v5.softmax(dim=-1)
        v7 = torch.nn.functional.dropout(v6, p=0.2)
        v8 = torch.matmul(v7, v3)
        o = self.conv_o(v8)
        return o

# Initializing the model
m = Model((0, 80, 240), (0, 80, 120), (0, 80, 120))

# Inputs to the model
batch_size, channels, height, width = 10, 80, 60, 70
x1 = torch.randn(batch_size, channels, height, width)
x2 = torch.randn(batch_size, channels, height, width)
