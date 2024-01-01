
class TransformerEncoderLayer(torch.nn.Module):
	def __init__(self, embedding_dim, num_heads,  dim_feedforward=2048, dropout=0.1, activation = 'ReLU'):
		super().__init__()
		self.dim_feedforward = dim_feedforward
		self.self_attn = MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
		# Implementation of Feedforward model
		forward_layers = []
		forward_layers.append(torch.nn.Linear(embedding_dim, dim_feedforward))
		# if activation == 'GELU':
		#     forward_layers.append(torch.nn.GELU())
		# else:
		#     forward_layers.append(torch.nn.ReLU())
		forward_layers.append(torch.nn.Dropout(dropout))
		forward_layers.append(torch.nn.Linear(dim_feedforward, embedding_dim))
		self.linear1 = torch.nn.Sequential(*forward_layers)
		
		self.norm1 = torch.nn.LayerNorm(embedding_dim)
		self.norm2 = torch.nn.LayerNorm(embedding_dim)
		self.dropout1 = torch.nn.Dropout(dropout)
		self.dropout2 = torch.nn.Dropout(dropout)
	
	def forward(self, src, src_mask=None, src_key_padding_mask=None):
		 # This is how you can build a nested NN module
		 attn_output, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask)

		 # This is how you merge two layers: LayerNorm + linear + Dropout
		 out1 = self.norm1(src + self.dropout1(attn_output))
		 # This is how you can apply a non-linear activation here
		 linear1_output = self.linear1(out1)
	    
		 # This is how you merge two layers: LayerNorm + linear + Dropout
		 out2 = self.norm2(linear1_output + self.dropout2(linear1_output))
		 
		 return out2

# Initializing the model
m = TransformerEncoderLayer(embedding_dim=64, num_heads=2)

# Inputs to the model
x1 = torch.randn(4, 4, 64)
src_mask = torch.randn(4, 4, 4) > 0
