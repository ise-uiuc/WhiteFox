
class Model(torch.nn.Module):
    def __init__(self, query_dim, key_dim=None, value_dim=None, scale_factor=None, dropout_p=None):
        super().__init__()
        if key_dim is None:
            key_dim = 0
        if value_dim is None:
            value_dim = 0
        
        self.W_query = torch.nn.Linear(query_dim, self.embed_dim)
        if scale_factor is None:
            scale_factor = torch.Tensor([math.sqrt(query_dim)])
        if dropout_p is None:
            dropout_p = 0.1
            
        self.p_dropout = dropout_p
        self.scale_factor = scale_factor

    def forward(self, query, key, value, mask=None):
        query = self.W_query(query)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)

        if mask is not None:
            assert mask.shape == scaled_qk.shape, "mask should be %s but got %s"% (scaled_qk.shape, mask.shape)
            mask = mask.float()
            scaled_qk = scaled_qk.masked_fill(mask == 0, -1e32)

        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.p_dropout)
        output = dropout_qk.matmul(value)

        return output, dropout_qk
 
# Initializing the model
m = Model(query_dim, key_dim, value_dim, scale_factor, dropout_p)

# Inputs to the model
query = torch.randn(2, 3, query_dim)
key = torch.randn(2, 4, key_dim)
value = torch.randn(2, 4, value_dim)
