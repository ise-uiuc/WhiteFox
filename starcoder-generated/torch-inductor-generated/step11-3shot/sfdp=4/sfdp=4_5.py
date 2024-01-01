
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        embedding_dim = 32
        heads = 2
        dropout_rate = 0.1
        # Compute the number of attention heads
        attention_head_dim = int(embedding_dim / heads)
        # Compute the input dimension per head
        self.attention_head_dim = attention_head_dim
 
        # Set up the transformers
        self.self_attention = MultiHeadedSelfAttention(embedding_dim=embedding_dim, num_heads=heads)
        self.ffn = FFN(embedding_dim=embedding_dim, hidden_dim=embedding_dim * 2)
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-12)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-12)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
 
    def forward(self, x, attention_mask=None):
        # Apply the self-attention layer
        output = self.layernorm1(x)
        output = self.self_attention(query=output, attention_mask=attention_mask)
        output = self.dropout1(output)
        x = x + output
        # Apply the feed-forward layer
        output = self.layernorm2(x)
        output = self.ffn(output)
        output = self.dropout2(output)
        x = x + output
 
        return x
 
# Number of sequences
N = 32 # 32
# Dimension of the feature vector of each token of the sequence
D = 32 # 32
# Initialize the query tensor
query = torch.randn(N, D)
# Initializer the key and value tensors
key = torch.randn(N, D)
value = torch.randn(N, D)
 
# For training, the attention mask is a tensor that masks out all positions that should be attended to according to the mask value (0 indicates that the position is not relevant; 1 indicates it is relevant). 
attention_mask = torch.randint(low=0, high=2, size=[N, N]) # A matrix of 0s and 1s
# For inference, we usually choose an attention mask that has 1s in all the elements.
attention_mask = torch.ones([N, N]) # A matrix of 1s
 
# Initializing the module
m = Model()
 
# Inputs to the model
x = query
