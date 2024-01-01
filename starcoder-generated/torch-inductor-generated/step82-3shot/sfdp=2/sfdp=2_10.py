
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads, feedforward_dim, embedding_dim, dropout_p):
        super().__init__()
        self.q_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self.k_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        self.v_linear = torch.nn.Linear(embedding_dim, embedding_dim)
        
        self.dropout = torch.nn.Dropout(dropout_p)
        
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.embedding_dim / self.num_attention_heads)
        
        self.projection_linear = torch.nn.Linear(embedding_dim, embedding_dim)
 
    def transpose_for_scores(self, x):
        new_tensor_shape = x.size()[:-1] + \
            (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_tensor_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x, mask=None):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            q = masked_fill(q, mask, -1e6)
            v = masked_fill(v, mask, -1e6)
            k = masked_fill(k, mask, -1e6)
        
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        
        qk = torch.matmul(q, k)
        qk = qk / np.sqrt(self.attention_head_size)
        scaled_qk = qk
        
        softmax_qk = scaled_qk.softmax(dim=-1)
        
        dropout_qk = self.dropout(softmax_qk)
        
        output = dropout_qk.matmul(v)
        
        output = output.permute(0, 2, 1, 3)
        new_tensor_shape = output.size()[:-2] + (self.embedding_dim,)
        output = output.reshape(*new_tensor_shape)
        
        output = self.projection_linear(output)
        
        return output, softmax_qk

# Initializing the model
num_attention_heads = 4
embedding_dim = 128
feedforward_dim = 512
dropout_p = 0.3
m = Model(num_attention_heads, feedforward_dim, embedding_dim, dropout_p)

# Inputs to the model
x = torch.randn(128, 128)
mask = torch.eye(128)
__output__, __softmax__ = m(x, mask=mask)

