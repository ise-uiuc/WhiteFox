
class Model(torch.nn.Module):
    __constants__ = ['dropout_p', 'inv_scale_factor']

    def __init__(self):
        super(Model, self).__init__()
        self.dropout_p = config.dropout_p
        self.conv = torch.nn.Conv1d(
            config.hidden_size, 10, kernel_size=3, stride=3)
        self.att_dropout = torch.nn.Dropout(config.dropout_p)
        self.att_conv = torch.nn.Conv1d(9, config.num_heads, kernel_size=2, stride=2)
        self.attn = Attention(config)
        self.inv_scale_factor = math.sqrt(float(config.hidden_size))

    def forward(self, value, key):
        v1 = self.conv(value)
        v2 = self.att_conv(key)
        v3 = v1 * v2
        v4 = self.attn(v3)
        v5 = v3 * v4
        v6 = torch.nn.functional.relu(v5)
        v7 = torch.nn.functional.softmax(v6)
        v8 = self.att_dropout(v7)
        v9 = v1.matmul(v8.transpose(-1, -2))
        return v9

# Initializing the model
m = Model()

# Inputs to the model
value = torch.randn(2, 5, 2, 3)
key = torch.randn(3, 5, 4, 3)

