
class Model(torch.nn.Module):
    def __init__(self, batch_size, seq_len, num_head):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_head = num_head

    def forward(self, q, k, v):
        q = q.reshape(self.batch_size * self.seq_len, self.num_head, -1)
        k = k.reshape(self.batch_size * self.seq_len, self.num_head, -1)
        v = v.reshape(self.batch_size * self.seq_len, self.num_head, -1)

        # q, k, v: batch_size * seq_len, num_head, hidden_dim
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        qk = torch.matmul(q, k.transpose(1, 2))
        scaled_qk = qk.div(64 ** 0.5)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)

        output = dropout_qk.matmul(v)
        output = output.transpose(1, 0)
        output = output.reshape(self.num_head, self.batch_size, self.seq_len, -1)
        output = output.permute(1, 2, 0, 3)

        return output

# Initializing the model
m = Model(1, 512, 8)

# Inputs to the model
q = torch.randn(512, 8, 32)
k = torch.randn(512, 8, 32)
v = torch.randn(512, 8, 64)
