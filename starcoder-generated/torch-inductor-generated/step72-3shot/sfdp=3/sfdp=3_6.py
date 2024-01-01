
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(input_vocab_size, embedding_dim)
        self.transformer = torch.nn.Transformer(num_encoder_layers, num_decoder_layers, embedding_dim, num_head, dim_linear_block, dropout_p)
 
    def forward(self, x1, x2):
        # Embedding the encoder input
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # Converting the shape of the tensors from (seq_len, batch_size, embedding_dim) to (seq_len*batch_size, embedding_dim)
        x1 = torch.reshape(x1, (-1, x1.shape[2]))
        x2 = torch.reshape(x2, (-1, x2.shape[2]))
 
        # Applying the transformer
        output = self.transformer(x1, x2)

# Initializing the model
input_vocab_size = 100
output_vocab_size = 100
embedding_dim = 64
num_head = 8
dim_linear_block = 128
total_key_depth = 1024
total_value_depth = 1024
input_len = 100
dropout_p = 1.0
num_encoder_layers = 6
num_decoder_layers = 6
m = Model()

# Inputs to the model
x1 = torch.randint(0, 100, (input_len, 1))
x2 = torch.randint(0, 100, (input_len, 1))
