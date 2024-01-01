
embedding_size = 32
num_heads = 8
m = nn.MultiheadAttention(embedding_size, num_heads)
 
batch_size = 1
sequence_length = 10
attn_mask = torch.tril(torch.ones((sequence_length, sequence_length)))
xquery = torch.randn(batch_size, sequence_length, embedding_size)
xkey = torch.randn(batch_size, sequence_length, embedding_size)
xvalue = torch.randn(batch_size, sequence_length, embedding_size)
__, xattn_weight = m(xquery, xkey, xvalue, attn_mask)
