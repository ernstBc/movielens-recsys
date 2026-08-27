import torch
from torch import nn
from torch.nn import functional as F



class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, user_embedding_dim, item_embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(n_users, user_embedding_dim)
        self.item_embedding = nn.Embedding(n_items, item_embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)


    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        user_bias = self.user_bias(user_indices).squeeze()
        item_bias = self.item_bias(item_indices).squeeze()

        dot_product = (user_embedded * item_embedded).sum(dim=1)
        return dot_product + user_bias + item_bias


class DeepMatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, user_embedding_dim, item_embedding_dim, hidden_dims:list[int], dropout_rate:float) -> None:
        super().__init__()

        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=user_embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=item_embedding_dim)

        fc = [nn.Linear(in_features=user_embedding_dim + item_embedding_dim, out_features=hidden_dims[0]),
                                      nn.ReLU(), 
                                      nn.Dropout(dropout_rate)]

        for prev_hidden, pos_hidden in zip(hidden_dims[:-1], hidden_dims[1:]):
            fc.append(nn.Linear(in_features=prev_hidden, out_features=pos_hidden))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(dropout_rate))

        fc.append(nn.Linear(in_features=hidden_dims[-1], out_features=1))
        self.fc = nn.Sequential(*fc)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


    def forward(self, user_id, item_id):
        user_embedded = self.user_embedding(user_id)
        item_embedded = self.item_embedding(item_id)

        x = torch.cat([user_embedded, item_embedded], dim=-1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    n_users = 20 + 1
    n_items = 100 + 1
    user_emb_size = 32
    item_embd_size = 32
    user_id_batch_example = torch.randint(0, n_users, size=(32, ))
    item_id_batch_example = torch.randint(0, n_items, size=(32, ))

    mf = MatrixFactorization(n_users=n_users, 
                             n_items=n_items, 
                             user_embedding_dim=user_emb_size, 
                             item_embedding_dim=item_embd_size)

    deep_mf = DeepMatrixFactorization(n_users=n_users,
                                      n_items=n_items,
                                      user_embedding_dim=user_emb_size,
                                      item_embedding_dim=item_embd_size,
                                      hidden_dims=[64,32,16],
                                      dropout_rate=0.5)

    mf_output = mf(user_id_batch_example, item_id_batch_example)
    print(mf_output.shape)
    print(mf_output)

    deep_mf_output = deep_mf(user_id_batch_example, item_id_batch_example)
    print(deep_mf_output.shape)