
class SelfAttention(torch.nn.Module):
    def __init__(self, head_num, head_dim):
        super().__init__()
        
        self.head_num = head_num
        self.head_dim = head_dim
        
        self.attention_scale_factor = 1.0 / math.sqrt(head_dim)
        self.dropout_p = 0.2

        self.query_projection = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_projection = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value_projection = torch.nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, embedding_channels = x.shape
        
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        
        qk = query @ key.transpose(-2, -1) * self.attention_scale_factor
        scaled_qk = qk
        softmax_qk = torch.exp(softmax(scaled_qk, dim=-1))
        # Dropout is not applied here since it is applied in the EmbeddingPostprocessor of BERT
        dropout_qk = softmax_qk
        output = dropout_qk @ value

        output = output.reshape([batch_size, seq_len, embedding_channels])
        return output
