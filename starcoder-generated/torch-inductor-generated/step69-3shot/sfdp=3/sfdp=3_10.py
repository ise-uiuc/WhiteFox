
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(20, 23, 512, 64))
        self.key = torch.nn.Parameter(torch.randn(20, 23, 64, 512))
        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.dropout_p = 0.1
 
    def forward(self, x):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output

# Initializing the model
batch_size = 8
seq_len = 200
head_num = 99
head_size = 1024
key_dim = 64
value_dim = 512
x = torch.randn(batch_size, seq_len, head_num * head_size)
