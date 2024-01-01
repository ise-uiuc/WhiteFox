
class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config['embedding_dim']
        self.nb_heads = config['nb_heads']
        self.nb_blocks = config['nb_blocks']

        self.scale_factor = 1.0 / (self.embedding_dim ** 0.5)
        self.dropout_p = 0.1

        self.query = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.key = torch.nn.Linear(self.embedding_dim, self.embedding_dim)
        self.value = torch.nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, inputs):
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        
        res = []
        res += query.unsqueeze(1).split([self.nb_heads[i] for i in range(self.nb_blocks)], dim=-1)
        res += key.split([self.nb_heads[i] for i in range(self.nb_blocks)], dim=-1)
        res += value.split([self.nb_heads[i] for i in range(self.nb_blocks)], dim=-1)
                
        out, attn = res[0], res[1:]
        for i in range(self.nb_blocks):
            temp, attn = out, attn[i]
            out, attn = self.attention(out, attn)
            out = out.add(temp)
 
        outputs = out.split([self.nb_heads[i] for i in range(self.nb_blocks)], dim=-1)
 
        res = outputs[0]
        for i in range(1, len(outputs)):
            res += out[i]

        return res

    def attention(self, query, key):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk.matmul(value), dropout_qk
