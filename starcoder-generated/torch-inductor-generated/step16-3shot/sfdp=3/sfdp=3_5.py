
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_segments=4, num_heads=2):
        super().__init__()
        self.dropout_p = 0.2
        self.scale_factor = np.power(hidden_size, -0.5)
        hidden_size_per_head = int(hidden_size / num_heads)
        self.q_key_value = torch.nn.Linear(hidden_size, (2 * num_segments + 7) * num_heads * hidden_size_per_head, bias=False)
        self.dropout = torch.nn.Dropout(p=0.2)
 
    def forward(self, query, key, value):
        q_key_value = self.q_key_value(query)
        q_key_value = q_key_value.reshape(q_key_value.shape[0], q_key_value.shape[1], num_segments + 3, num_heads,
                                          hidden_size_per_head)
        q_key_value = q_key_value.permute(0, 2, 1, 3, 4)
        q, k, v = q_key_value[:, 0, :, :, :], q_key_value[:, 1, :, :, :], q_key_value[:, 2, :, :, :]
        q, k = q.reshape(-1, q.shape[-2], q.shape[-1]), k.reshape(-1, k.shape[-2], k.shape[-1])
        scaled_qk = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, v.reshape(-1, v.shape[-2], v.shape[-1])).reshape(dropout_qk.shape[0], dropout_qk.shape[1], num_segments, num_heads,
                                                                      hidden_size_per_head).transpose(1, 2)
        output = output.reshape(output.shape[0], output.shape[1], num_heads * hidden_size_per_head)
        return output

# Initializing the model
m = Model(hidden_size=16)

# Inputs to the model
query = torch.randn(1, 2, 4)
key = torch.randn(1, 10, 4)
value = torch.randn(1, 10, 4)
