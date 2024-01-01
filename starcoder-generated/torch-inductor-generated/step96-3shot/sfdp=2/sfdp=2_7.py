
class ScaledDotProductAttention(object):
    def __init__(self, dropout_p):
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value, mask):
        dot_product_qk = torch.matmul(query, key.transpose(-2, -1))
        inv_sqrt_dk = 1 / math.sqrt(query.size(-1))
        scaled_qk = dot_product_qk * inv_sqrt_dk[None, None, :]
        if mask is not None:
            scaled_qk += mask
        attention_weights = scaled_qk.softmax(-1)
        attention_dropped = F.dropout(attention_weights, p=self.dropout_p)
        attention_outputs = torch.matmul(attention_dropped, value)
        return attention_outputs
 
class MultiHeadAttention(object):
    def __init__(self, dropout_p, num_heads):
        self.dropout_p = dropout_p
        self.num_heads = num_heads
        self.depth = None
 
    def forward(self, query, key, value, mask):
        if self.depth is None:
            self.depth = (key.size(-1) // self.num_heads,
                          query.size(-1) // self.num_heads)
        self._validate_inputs(query, key, value, mask)
        batch_size = query.size(0)
        query, key, value = map(lambda x: x.view(batch_size, -1, self.num_heads, self.depth[0]),
                                  (query, key, value))
        x = ScaledDotProductAttention(self.dropout_p)
        return x(query, key, value, mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1))
 
    def _validate_inputs(self, query, key, value, mask):
        if query.dim()!= 3:
            raise RuntimeError("Query must be a 3D tensor.")
        if key.dim()!= 3:
            raise RuntimeError("Key must be a 3D tensor.")
        if value.dim()!= 3:
            raise RuntimeError("Value must be a 3D tensor.")
        if query.size(0)!= key.size(0):
            raise RuntimeError("Found quer batch size {}, expected {}."
                              .format(query.size(0), key.size(0)))
        if query.size(2)!= key.size(2):
            raise RuntimeError("Found query embedding size {}, expected {}."
                              .format(query.size(2), key.size(2)))
        if key.size(0)!= value.size(0):
            raise RuntimeError("Found key batch size {}, expected {}."
                              .format(key.size(0), value.size(0)))
        if key.size(2)!= value.size(2):
            raise RuntimeError("Found key embedding size {}, expected {}."
                              .format(key.size(2), value.size(2)))
        if mask is not None:
            if mask.dim()!= 3:
                raise RuntimeError("Mask must be a 3D tensor.")
            if mask.size(0)!= query.size(0):
                raise RuntimeError("Found mask batch size {}, expected {}."
                                  .format(mask.size(0), query.size(0)))
            if mask.size(2)!= query.size(2):
                raise RuntimeError("Found mask embedding size {}, expected {}."
                                  .format(mask.size(2), query.size(2)))
            if (mask.byte() & 1).any():
                raise RuntimeError("Entries in mask should be 0 or 1.")
 
class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout_p):
        super().__init__()
        self.enc_head_lnrm = MultiHeadAttention(dropout_p=dropout_p, num_heads=num_heads)
        self.enc_lnrm1 = torch.nn.LayerNorm(normalized_shape=dim)
        self.enc_act1 = torch.nn.ReLU()
 
        self.enc_head_lnrm2 = MultiHeadAttention(dropout_p=dropout_p, num_heads=num_heads)
        self.enc_lnrm2 = torch.nn.LayerNorm(normalized_shape=dim)
        self.enc_act2 = torch.nn.ReLU()
 
    def forward(self, x, mask):
        att_out = self.enc_head_lnrm(x, x, x, mask)
        att_out += x
        x = self.enc_lnrm1(att_out)
        x = self.enc_act1(x)
        att_out = self.enc_head_lnrm2(x, x, x, mask)
        att_out += x
        return self.enc_lnrm2(att_out), mask
 
class Model(torch.nn.Module):
    def __init__(self, d, d_ff, num_heads, dropout_p=0.1, num_layers=2):
        super().__init__()
        self.scale_factor = math.sqrt(d_ff)
        self.input_layer = torch.nn.Linear(in_features=d, out_features=d_ff)
        self.input_layer_norm = torch.nn.LayerNorm(normalized_shape=d_ff)
 
        sub_layers = []
        for _ in range(num_layers):
            sub_layers += [TransformerEncoderBlock(dim=d_ff,
                                                   num_heads=num_heads,
                                                   dropout_p=dropout_p)]
        self.sub_layers = torch.nn.ModuleList(sub_layers)
 
        self.output_layer = torch.nn.Linear(in_features=d_ff, out_features=d)
 
    def forward(self, x, mask):
        x = self.input_layer(x)
        x = self.input_layer_norm(x)
        x = F.relu(x)
        x = x.permute(1, 0, 2)
        mask = mask.permute(1, 0)
 
        for sublayer in self.sub_layers:
            x, mask = sublayer(x, mask)
 
        output = x.permute(1, 0, 2)
        return self.output_layer(output)

# Initializing the model
m = Model(d=2, d_ff=64, num_heads=4)
 
# Inputs to the model
x = torch.tensor([[1., 1.], [1., 1.]])
mask = torch.tensor([[1., 0.], [0., 1.]])
