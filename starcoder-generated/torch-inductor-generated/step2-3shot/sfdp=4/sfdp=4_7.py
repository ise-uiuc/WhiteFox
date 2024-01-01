
seq_len, batch_size, num_head, dim_head, dim_v = torch.Size([3]), torch.Size([2]), torch.Size([2]), torch.Size([8]), torch.Size([8])
mha = torch.nn.MultiheadAttention(num_head, dim_head)
torch.manual_seed(0)
x1 = torch.randn(seq_len, batch_size, dim_v)
x2 = torch.randn(seq_len, batch_size, dim_v)
x3 = torch.randn(1, 2, 2)
mha_input = (x1, x2, x3)
mha_output = mha(*mha_input)

