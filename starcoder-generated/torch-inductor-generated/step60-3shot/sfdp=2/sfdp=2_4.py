
class Model(torch.nn.Module):
    def __init__(self, num_heads=8, dropout_p=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.inv_scale_factor = np.power(self.num_heads, -0.5)
        self.qkv = torch.nn.Linear(32, 32 * 3)
    
    def forward(self, inputs):
        q, k, v = torch.chunk(self.qkv(inputs).view(
            inputs.size(0), 3, self.num_heads, -1), 3, dim=1)
        q *= self.inv_scale_factor
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        # We should return the shape (batch_size, seq_len, 32)
        return output.transpose(1, 2)

# Initializing the model
m = Model()

# Inputs to the model
inputs = torch.randn(1, 32, 64)
