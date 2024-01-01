
class Model(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout_p=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.dropout_p = dropout_p
        self._reset_parameters()
 
    def _reset_parameters(self):
        nn.init.xavier_normal_(self.query.weight)
        nn.init.xavier_normal_(self.key.weight)
        nn.init.xavier_normal_(self.value.weight)
 
    def forward(self, q, k, v, attn_mask=None):
        bs = q.size(0)
 
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(1).repeat(1,self.nhead,1,1).unsqueeze(1)
        q = self.query(q).view(bs,self.nhead,self.d_model//self.nhead,-1)
        k = self.key(k).view(bs,self.nhead,self.d_model//self.nhead,-1)
        v = self.value(v).view(bs,self.nhead,self.d_model//self.nhead,-1)
 
        q = q * self.scale
        q = q.transpose(2, 3)
        attn_weight = torch.bmm(q, k.transpose(2,3))
        attn_weight = attn_weight + attn_mask
        attn_weight = attn_weight.masked_fill(attn_weight == 0, -10e150)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = F.dropout(attn_weight, p=self.dropout_p, training=self.training)
        output = torch.bmm(attn_weight, v)
 
        output = output.transpose(2,1).view(bs,-1)
        output = self.output(output)
 
        return output
 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 11
        self.n_head = 1
        self.n_layer = 1
        self.num_attention_heads = self.n_head
        d_model = self.embedding_dim
        self.dim_feedforward = 32
        self.dropout_rate = 0.1
        self.embedding_dropout = 0.1
        self.embedding = nn.Embedding(32, d_model)
        self.transformer = nn.Transformer(d_model, nhead=self.n_head, num_encoder_layers=self.n_layer,
                                          num_decoder_layers=self.n_layer, dim_feedforward=self.dim_feedforward, dropout=self.dropout_rate)
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(96, 26)
 
    def forward(self, src, tgt):
        if self.embedding_dropout == 0:
            embedded_src = self.embedding(src)
            embedded_tgt = self.embedding(tgt)
        else:
            dropout = nn.Dropout(self.embedding_dropout)
            embedded_src = dropout(self.embedding(src))
            embedded_tgt = dropout(self.embedding(tgt))
        v = self.transformer(embedded_src, embedded_tgt)
        v = torch.cat((v, embedded_src, embedded_tgt), 1)
        v = v.view(v.shape[0], -1)
        v = self.fc(v)
        return torch.matmul(F.softmax(x), y)
 