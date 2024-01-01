
class Rearrange(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        dim_0 = x.shape[0]
        dim_1 = x.shape[1]
        dim_2 = x.shape[2]
        dim_3 = math.ceil(dim_2 / 32)
        x = x.reshpae(dim_0, dim_1, 32, dim_3)
        x = x.transpose(-1, -2)
        dim_1 = x.shape[-1]
        x = x.reshape(dim_0, dim_1, -1)
        dim_0 = x.shape[0]
        dim_1 = x.shape[1]
        dim_2 = x.shape[2]
        x = x.reshape(dim_0, dim_1, dim_2, 4)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(dim_0 * 4, dim_1, dim_2)
        return x
class Reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        dim_0 = x.shape[0]
        dim_1 = x.shape[1]
        dim_2 = x.shape[2]
        x = torch.reshape(x, (dim_0, 32, -1))
        dim_1, dim_2 = x.shape[-2:]
        for i in range(4):
            x_i = x[:, :, dim_1 * i: dim_1 * (i + 1)]
            x_i = torch.reshape(x_i, (dim_0, 32, 4, -1))
            x_i = torch.transpose(x_i, 1, 2)
            x_i = x_i.reshape(dim_0, 4, dim_1 * 32, -1)
            y_i = x_i[:, i, :, :]
            y_i = y_i.reshape(dim_0, dim_1, -1)
            if i == 0:
                y = y_i
            else:
                y = torch.cat((y, y_i), dim=1)
        y = y.reshape(dim_0 * 4, dim_1, -1)
        return y
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = 8
        self.seq_len = 512
        self.dim = 64 // self.heads
        self.linear_0 = torch.nn.Linear(self.dim * 8, self.dim)
        self.linear_1 = torch.nn.Linear(self.dim, self.dim * 8)
        self.reshape = Reshape()
        self.rearrange = Rearrange()
    def forward(self, x):
        batch_size = q.shape[0]
        q = self.reshape(q)
        k = self.reshape(k)
        v = self.reshape(v)
        h = self.heads
        dim = self.dim
        q = q.reshape(batch_size, h, -1, dim)
        k = k.reshape(batch_size, h, -1, dim)
        v = v.reshape(batch_size, h, -1, dim)
        attention_scores = torch.einsum('bthe,btje->bhts', q, k) # einsum
        attention_scores = attention_scores / math.sqrt(dim)
        attention_scores = attention_scores + attn_mask
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.view(batch_size, *attention_weights.shape[1:])
        attention_weights = torch.nn.functional.dropout(attention_weights, p=0.1, training=True)
        context = torch.einsum('bhts,btje->bthe', attention_weights, v) # einsum
        context = self.rearrange(context)
        x = torch.nn.functional.activation(self.linear_0(x))
        x = torch.nn.functional.dropout(x, p=0.1, training=True)
        x = x + context
        x = self.rearrange(x)
        x = torch.nn.functional.activation(self.linear_1(x))
        return x
# Inputs to the model
q = torch.randn(1, 8192, 256)
k = torch.randn(1, 8192, 256)
v = torch.randn(1, 8192, 256)
attn_mask = torch.randn(1, 1, 8192, 8192)
