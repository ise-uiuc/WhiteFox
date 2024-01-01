
class Model(torch.nn.Module):
    def __init__(self,
                 n_head,
                 d_model,
                 dropout_p=0,
                 bias=True,
                 pad=False):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.dropout_p = dropout_p
        self.bias = bias
        self.pad = pad
        self.inner_dims = d_model // n_head
        self.proj_factor = self.inner_dims ** -0.5
        self.proj_value = torch.nn.Linear(self.inner_dims, self.inner_dims, bias=bias)
        self.proj_query = torch.nn.Linear(self.inner_dims, self.inner_dims, bias=bias)
        self.proj_out = torch.nn.Linear(self.d_model, self.d_model, bias=bias)

    def forward(self, x):
        n_batch, len_seq, _ = x.size() 
        x_reshape = x.view(n_batch*len_seq, self.n_head, self.inner_dims)
        x_query, x_key, x_value = self.proj_query(x_reshape), self.proj_key(x_reshape), self.proj_value(x_reshape)
        x_query = x_query.view(n_batch*len_seq, self.n_head*self.inner_dims)
        x_query = x_query.view(n_batch, len_seq*self.n_head, self.inner_dims)
        x_key = x_key.view(n_batch*len_seq, self.n_head*self.inner_dims)
        x_value = x_value.view(n_batch*len_seq, self.n_head*self.inner_dims)
        x_attn = x_query.view(n_batch*len_seq*self.n_head, self.inner_dims, 1)
        x_prod = torch.matmul(x_attn, x_key.transpose(1,2).view(n_batch*len_seq*self.n_head, self.inner_dims, self.n_head*self.inner_dims))
        x_prod = x_prod.view(n_batch, len_seq*self.n_head, self.inner_dims, self.n_head*self.inner_dims)
        x_prod = x_prod / self.proj_factor
        x_prod = torch.matmul(x_prod, x_value.view(n_batch, len_seq*self.n_head, self.inner_dims, self.inner_dims).transpose(2,3))
        x_prod = x_prod.view(n_batch*len_seq*self.n_head, self.inner_dims)
        x_out = torch.nn.functional.dropout(x_prod, 1-self.dropout_p)
        x_out = x_out.view(n_batch, len_seq*self.n_head, self.inner_dims)
        x_out = x_out.view(n_batch, len_seq, self.d_model)
        output = self.proj_out(x_out)
        return output

# Initializing the model
m = Model(n_head, d_model, dropout_p)

# Inputs to the model
x1 = torch.randn(n_batch, len_seq, d_model)
