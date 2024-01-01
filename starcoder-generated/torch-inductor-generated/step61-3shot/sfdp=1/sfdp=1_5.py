
class Model(torch.nn.Module):
    def __init__(self, batch, embedding, num_head):
        super().__init__()
        self.num_head = num_head
        self.batch = batch
        self.embedding = embedding
        self.q_linear = torch.nn.Linear(self.embedding, self.embedding)
        self.k_linear = torch.nn.Linear(self.embedding, self.embedding)
        self.v_linear = torch.nn.Linear(self.embedding, self.embedding)
 
    def forward(self, q, k, v, pos, training=True):
        r_qk = self.mha(q, k, v, pos)
        r_qk = r_qk.transpose(-2, -1)
        a = r_qk / math.sqrt(self.embedding / self.num_head)
        a = a.softmax(dim=-1)
        #a_dropout = torch.nn.functional.dropout(a)
        r = torch.matmul(a, v)
        return r
 
class mha(torch.nn.Module):   
    def __init__(self):
        super().__init__()
        self.transpose = torch.nn.functional.linear(A=q, B=k, bias=None)
    def forward(self, q, k, v, pos):
        q_key = torch.matmul(q, k.transpose(-2, -1))
        q_key = q_key.div(math.sqrt(self.embedding / self.num_head))
        q_key = q_key.softmax(dim=-1)
        q_key_dropout = torch.nn.functional.dropout(q_key)
        output = torch.matmul(q_key_dropout, v)
        return output

# Initializing the model
m = Model()

# Query, Key and Value
__batch__ = 4
__embedding__ = 128
__num_head__ = 2

q = torch.randn(self.batch*self.num_head, __num_head__, __embedding__)
k = torch.randn(self.q_len*self.num_head, __num_head__, __embedding__)
v = torch.randn(self.q_len*self.num_head, __num_head__, __embedding__)

# Pos
__max_len__ = 41
__pos_len__ = 41
self.pos = torch.tensor([[pos_i-pos_i%self.num_head for pos_i in pos_j] for pos_j in pos])
self.pos = self.pos.view(-1).to(m.device)
pos_encoding_tensor = torch.zeros(self.len, self.depth)
pos = torch.arange(self.len).unsqueeze(1).expand(self.len, self.depth).to(m.device)
__pos_encoding__ = torch.tensor([pos_i % self.depth for pos_i in pos])
pos_encoding_tensor[...] = torch.nn.functional.embedding(input=__pos_encoding__, weight=self.position_encoding_table)
pos_encoding_tensor[::self.num_head,...] = torch.nn.functional.embedding(input=__pos_encoding__, weight=self.position_encoding_table)
__pos_encoding__ = pos_encoding_tensor[range(self.len)]

# Inputs to the model
__query__ = q
__key__ = k
__value__ = v
__pos__ = __pos_encoding__
__training__ = True
r = m(query=__query__, key=__key__, value=__value__, pos=__pos__, training=__training__)

