
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads, head_size):
        super().__init__()
        self.dropout_p = 0.9
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        self.dropout = torch.nn.Dropout(self.dropout_p)

    def forward(self, query, key, value, inv_scale_factor):
        query = query.transpose(0, 1) # (num_heads, batch, sequence_length, embed_size_per_head) -> (batch, num_heads, sequence_length, embed_size_per_head)
        key = key.transpose(0, 1) # (num_heads, batch, sequence_length, embed_size_per_head) -> (batch, num_heads, sequence_length, embed_size_per_head)
        value = value.transpose(0, 1) # (num_heads, batch, sequence_length, embed_size_per_head) -> (batch, num_heads, sequence_length, embed_size_per_head)
        qk = torch.matmul(query, key.transpose(-2, -1)) # (batch, num_heads, sequence_length, sequence_length)
        scaled_qk = qk.div(inv_scale_factor) # (batch, num_heads, sequence_length, sequence_length)
        softmax_qk = scaled_qk.softmax(dim=-1) # (batch, num_heads, sequence_length, sequence_length)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # (batch, num_heads, sequence_length, sequence_length)
        output = dropout_qk.matmul(value) # (batch, num_heads, sequence_length, sequence_length)
        return output.transpose(0, 1) # (batch, num_heads, sequence_length, sequence_length) -> (num_heads, batch, sequence_length, sequence_length)
        
# Initializing the model
num_attention_heads = 12
head_size = 64
inv_scale_factor = 1.0 / (head_size ** 0.5)
m = Model(num_attention_heads, head_size)

# Inputs to the model
from transformers import BatchEncoding
batch = {"input_ids": torch.arange(512), "attention_mask": torch.ones(512)}
batch_attention_mask = BatchEncoding(batch, tensor_type="pt").attention_mask
inv_scale_factor = inv_scale_factor * batch_attention_mask
