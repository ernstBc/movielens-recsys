import torch 
from torch import nn



class MLPRec(nn.Module):
    def __init__(self, n_users:int, n_items:int, embedding_dim:int,n_hidden_layers:int=3, dropout_rate:float=0.2):
        super(MLPRec, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_dim, padding_idx=0)

        layers = [nn.Linear(embedding_dim*2, embedding_dim),
                  nn.ReLU(),
                  nn.Dropout(dropout_rate),
                  nn.BatchNorm1d(embedding_dim)]

        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(embedding_dim, embedding_dim),
                           nn.ReLU(),
                           nn.Dropout(dropout_rate),
                           nn.BatchNorm1d(embedding_dim)])

        self.layers = nn.Sequential(*layers)


        self._init_he_weights()


    def _init_he_weights(self):
        nn.init.kaiming_uniform_(self.user_embedding.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.item_embedding.weight, nonlinearity='relu')
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')


    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)

        vector = torch.cat([user_embedded, item_embedded], dim=-1)
        prediction = self.layers(vector)
        return prediction



class GMFLayer(nn.Module):
    def __init__(self, n_users:int, n_items:int, embedding_dim:int):
        super(GMFLayer, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=embedding_dim, padding_idx=0)

        self._init_he_weights()


    def _init_he_weights(self):
        nn.init.kaiming_uniform_(self.user_embedding.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.item_embedding.weight, nonlinearity='relu')


    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices).squeeze(1)
        item_embedded = self.item_embedding(item_indices).squeeze(1)
        prediction = torch.mul(user_embedded, item_embedded)
        return prediction


class NeuMF(nn.Module):
    def __init__(self, n_users:int, n_items:int, embedding_dim:int,n_hidden_layers:int, dropout_rate=0.2):
        super(NeuMF, self).__init__()
        self.mlp = MLPRec(n_users=n_users, n_items=n_items, embedding_dim=embedding_dim, n_hidden_layers=n_hidden_layers, dropout_rate=dropout_rate)
        self.gmf = GMFLayer(n_users=n_users, n_items=n_items, embedding_dim=embedding_dim)
        self.neumf_layer = nn.Linear(in_features=embedding_dim * 2,  out_features=1)


    def forward(self, user_indices, item_indices):
        mlp_prediction = self.mlp(user_indices, item_indices)
        gmf_prediction = self.gmf(user_indices, item_indices)
        mlp_fmg_preds = torch.cat([mlp_prediction, gmf_prediction], dim=-1)
        prediction = self.neumf_layer(mlp_fmg_preds)
        return prediction.squeeze()


if __name__=='__main__':
    n_users = 20 + 1
    n_items = 100 + 1
    embedding_dim = 64
    n_layers = 2
    user_id_batch_example = torch.randint(1, n_users-1, size=(32, ))
    item_id_batch_example = torch.randint(0, n_items-1, size=(32,))
    item_id_negsample_example = torch.randint(0, n_items-1, size=(32, ))

    neumf = NeuMF(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_hidden_layers=n_layers,
        dropout_rate=0.5
    )


    preds = neumf(user_id_batch_example, item_id_batch_example)
    print('Prediction shape', preds.shape)
    print('Single Predictions:', preds)