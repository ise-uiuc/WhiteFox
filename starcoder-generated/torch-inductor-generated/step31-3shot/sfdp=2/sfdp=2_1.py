
class Model(torch.nn.Module):
    def __init__(self, attention_hidden_size, num_attention_heads, dropout_p):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax()
 
    def forward(self, query, key, value, scale_factor=None, dropout_p=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk/scale_factor
        softmax_qk = self.softmax(scaled_qk, dim=-1)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(attention_hidden_size=512, num_attention_heads=4, dropout_p=0.5)

# Inputs to the model
query = torch.randn(15, 4, 512)
key = torch.randn(16, 4, 512)
value = torch.randn(16, 4, 512)
scale_factor = torch.scalar_tensor(1/math.sqrt(512))
