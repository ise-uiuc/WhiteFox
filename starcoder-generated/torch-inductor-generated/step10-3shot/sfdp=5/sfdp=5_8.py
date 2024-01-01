
class Transformer(nn.Module):
    def forward(self, query, key, value, attn_mask, dropout_p=0.0):
        bsz, len_q = query.size(0), query.size(1)
        bsz, len_k = key.size(0), key.size(1)
        bsz, len_v = value.size(0), value.size(1)
        _____ = torch.bmm(query, key.transpose(1, 2))
        query = query.view(bsz, len_q, 1, ______).repeat(1, 1, len_k, 1)
        key = key.view(bsz, 1, len_k, ______).repeat(1, len_q, 1, 1)
        dks = ______ + ______
        dks = dks / math.sqrt(_______.size(dim=-1))
        dks = dks + ______
        ______ = F.softmax(______, dim=-1)
        ______ = ______ * ______ # Add the attention mask to the softmax
        ______ = F.dropout(______, p=dropout_p, training=self.training)
        ______ = torch.bmm(______, value)
        ______ = ______.view(bsz, len_q, len_v)
        return _______

# Initializing the model
model = Transformer()

# Inputs to the model
query = torch.randn(8, 16, 128)
key = torch.randn(8, 64, 256)
value = torch.randn(8, 64, 16)
attn_mask = torch.tensor([[1, 0, 1], [1, 0, 0], [0, 1, 1]])
dropout_p = 0.25
