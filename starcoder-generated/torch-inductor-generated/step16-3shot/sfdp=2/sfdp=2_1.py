
class Model(torch.nn.Module):
    def __init__(self, dropout_p=0.5, n_heads=8):
        super().__init__()
        self.dropout_p = dropout_p
        self.n_heads = n_heads
 
    def forward(self, query, key, value, mask=None):
        scaled_qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.sqrt(torch.tensor(key.shape[-1], dtype=torch.float16))
        softmax_qk = scaled_qk / inv_scale_factor
        softmax_qk = softmax_qk.softmax(dim=-1)
        dropout = torch.nn.functional.dropout(softmax_qk, self.dropout_p)
        output = dropout.matmul(value)
        return output

# Initializing the model
m = Model()

# Input to the model
query = torch.randn(1, 32, 128)
key = torch.randn(1, 32, 256)
value = torch.randn(1, 32, 256)
