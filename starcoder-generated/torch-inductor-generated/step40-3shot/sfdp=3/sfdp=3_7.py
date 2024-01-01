
class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, h, dropout_p=0.1):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.dropout_p = dropout_p
        self.q_linear = nn.Linear(d_model, h * d_k)
        self.v_linear = nn.Linear(d_model, h * d_v)
        self.k_linear = nn.Linear(d_model, h * d_k)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(h * d_v, d_model)

    def forward(self, query, key, value):
        bs = query.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(query).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(query).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(query).view(bs, -1, self.h, self.d_v)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2) #(bs,sl,h,d_k)
        q = q.transpose(1,2) #(bs,sl,h,d_k)
        v = v.transpose(1,2) #(bs,sl,h,d_k)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, self.d_v, self.dropout_p)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
          .view(bs, -1, self.d_v* self.h)
        output = torch.tanh(self.out(concat))
        return output

bs = 1
sl = 128
d_k = d_v = 16
n = 1
h = 2
d_model = d_v * h
input_tensor = torch.randn(bs, sl, d_model)
model = MultiHeadAttention(d_k, d_v, h)
