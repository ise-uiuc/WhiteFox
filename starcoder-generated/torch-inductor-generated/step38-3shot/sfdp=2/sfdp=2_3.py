
class Model(torch.nn.Module):
    def __init__(self, num_heads=2, key_dim=128, use_bias=True):
        super().__init__()
        self.num_heads = num_heads
        for i in range(self.num_heads):
            setattr(self, 'q_{:d}h'.format(i+1), torch.nn.Linear(q_dim//self.num_heads, key_dim//self.num_heads, bias=use_bias))
            setattr(self, 'k_{:d}h'.format(i+1), torch.nn.Linear(k_dim//self.num_heads, key_dim//self.num_heads, bias=use_bias))
            setattr(self, 'v_{:d}h'.format(i+1), torch.nn.Linear(v_dim//self.num_heads, val_dim//self.num_heads, bias=use_bias))
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.output_linear = torch.nn.Linear(val_dim * num_heads, q_dim)
 
    def _reshape(self, x, batch_dims_to_flatten):
        for batch_dim in batch_dims_to_flatten:
            if batch_dim == 'none':
                pass
            else:
                assert x.size(batch_dim) == 1, \
                    'All batch dimensions except batch axis 0 should have size equal to 1, but found'\
                    'batch dimension of size {}'.format(x.size(batch_dim))
 
        x = x.contiguous()
        b, s, d = x.size()
        x = x.view(b, -1, s, d)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, s, -1)
        return x
 
    def forward(self, query, key, value):
        batch_dims_to_flatten = ['B3', '4']
        q = self._reshape(query, batch_dims_to_flatten)
        k = self._reshape(key, batch_dims_to_flatten)
        v = self._reshape(value, batch_dims_to_flatten)
 
        q_out = list()
        k_out = list()
        v_out = list()
        for i in range(self.num_heads):
            q_part = getattr(self, 'q_{:d}h'.format(i+1))(q)
            q_part = [q_part] * len(batch_dims_to_flatten)
            q_out.append(self._reshape(q_part[0], batch_dims_to_flatten))
            k_part = getattr(self, 'k_{:d}h'.format(i+1))(k)
            k_part = [k_part] * len(batch_dims_to_flatten)
            k_out.append(self._reshape(k_part[0], batch_dims_to_flatten))
            v_part = getattr(self, 'v_{:d}h'.format(i+1))(v)
            v_part = [v_part] * len(batch_dims_to_flatten)
            v_out.append(self._reshape(v_part[0], batch_dims_to_flatten))
 
        q_out,k_out,v_out = torch.cat(q_out, -1), torch.cat(k_out, -1), torch.cat(v_out, -1)
 
        qkv = torch.matmul(q_out, k_out.transpose(-2, -1))
        qkv = qkv.div(inv_scale_factor)
        qkv_softmax = qkv.softmax(dim=-1)
        qkv_out = self.dropout(qkv_softmax)
        output_linear_in = torch.matmul(qkv_out, v_out)
        output = self.output_linear(output_linear_in)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 3, d_model)
key = value = torch.randn(1, 3, d_model)
