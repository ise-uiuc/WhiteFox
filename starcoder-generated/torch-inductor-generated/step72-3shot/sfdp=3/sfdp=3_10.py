
class Model(torch.nn.Module):
    def forward(self, x0, x1, x2, x3, x4, x5):
        # Parameters
        drop_prob_0 = 0.0
        drop_prob_2 = 0.0
        hidden_size_0 = 4
        hidden_size_3 = 4
        w878b = np.load('weights/bert_encoder_transform.npz')
        w0 = np.load('weights/bert_embedding_weights.npz')
        w1 = np.load('weights/bert_embedding_weights.npz')
        bias = np.load('weights/bert_hidden_bias.npz')['variable_bert_encoder_transformer_encoder_layer_0_bert_attention_self_query_weight_grad']

        # Initializing the weights
        w0 = w0['variable_bert_encoder_embedding_lookup_table']
        w1 = w1['variable_bert_encoder_embedding_lookup_table']
        w2 = w878b['variable_bert_encoder_transform']
        w3 = w878b['variable_bert_encoder_transform']
        w5 = w878b['variable_bert_encoder_transform']

        # Initializing the bias
        v8 = np.reshape(bias, (8, 1))
        b0 = v8

        v9 = np.reshape(bias, (1, 8))
        b1 = v9

        # Model
        v0 = torch.nn.functional.embedding(x0, w0)
        v1 = torch.nn.functional.embedding(x1, w1)
        v2 = torch.nn.functional.dropout(v0, drop_prob_0)
        v3 = v2.matmul(w2) + b0
        v4 = v1 + v3
        v5 = torch.tanh(v4)
        v6 = torch.dropout(v5, drop_prob_2)
        v7 = v6.matmul(w3) + b1
        v8 = torch.tanh(v7)
        v9 = v8.matmul(w5)
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
x1 = torch.tensor([[4, 5, 6], [5, 6, 7]], dtype=torch.long)
x2 = torch.tensor([1.0, 2.0])
x3 = torch.tensor([1.0, 2.0])
x4 = torch.tensor([1.0, 2.0])
x5 = torch.tensor([1.0, 2.0])
