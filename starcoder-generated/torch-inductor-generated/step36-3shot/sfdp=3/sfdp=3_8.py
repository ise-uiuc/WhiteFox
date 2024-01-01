
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(x3)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=x4)
        output = dropout_qk.matmul(x5)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, encoder_seq_length, key_dim)
x2 = torch.randn(1, 1, decoder_seq_length, key_dim)
x3 = 1.0
x4 = 0.0
x5 = torch.randn(1, 1, decoder_seq_length, value_dim)
