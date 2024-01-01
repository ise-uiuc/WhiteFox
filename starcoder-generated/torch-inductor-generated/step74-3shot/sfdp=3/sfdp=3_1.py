
class Model(torch.nn.Module):
    def forward(self, q, k, v, scale_factor=1. / math.sqrt(512), dropout_p=0.3):
        m2 = torch.matmul(q, k.transpose(-2, -1))
        m3 = m2 * scale_factor
        m4 = torch.nn.functional.softmax(m3, dim=-1)
        m5 = torch.nn.functional.dropout(m4, p=dropout_p)
        output = torch.matmul(m5, v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, 512)
k = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, 512)
v = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, 512)
