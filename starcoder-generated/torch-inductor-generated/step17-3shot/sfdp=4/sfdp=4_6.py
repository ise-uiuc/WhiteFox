
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, q, k, v, attention_mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + attention_mask
        attention_weight = torch.softmax(qk, dim=-1)
        output = attention_weight @ v
        return output

# Initializing the model
batch_size = 1
q_seq_len = 32
k_seq_len = 32
v_seq_len = 32
head_num = 8
d_model = 128
q = torch.randn(batch_size * head_num, q_seq_len, d_model // head_num)
k = torch.randn(batch_size * head_num, k_seq_len, d_model // head_num)
v = torch.randn(batch_size * head_num, v_seq_len, d_model // head_num)
attention_mask = torch.rand(batch_size, 1, q_seq_len, k_seq_len) < 0.5
m = Model()
