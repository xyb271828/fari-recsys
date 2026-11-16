"""Two-tower PyTorch retrieval with an in-batch sampling correction."""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def tensor_contexts(rows, history_size=16):
    histories = np.full((len(rows), history_size), -1, dtype=np.int64)
    for index, row in enumerate(rows):
        histories[index, :len(row[3])] = row[3]
    return (
        torch.tensor([row[0] for row in rows], dtype=torch.long),
        torch.tensor([row[2] for row in rows], dtype=torch.long),
        torch.from_numpy(histories),
    )


class TwoTower(nn.Module):
    def __init__(self, data, width=32, use_user_id=True):
        super().__init__()
        self.use_user_id = use_user_id
        _, categories = np.unique(data.catalog.categories, return_inverse=True)
        text = TfidfVectorizer(max_features=128).fit_transform(data.catalog.titles)
        self.register_buffer("categories", torch.from_numpy(categories))
        self.register_buffer("text", torch.tensor(text.toarray(), dtype=torch.float32))
        self.user_id = nn.Embedding(data.users + 1, width, padding_idx=0)
        self.item_id = nn.Embedding(data.catalog.size + 1, width, padding_idx=0)
        self.category = nn.Embedding(len(set(categories)), 12)
        self.hour = nn.Embedding(4, 8)
        self.text_projection = nn.Linear(self.text.shape[1], 16)
        self.user_net = nn.Sequential(nn.Linear(2 * width + 8, 64), nn.ReLU(), nn.Linear(64, width))
        self.item_net = nn.Sequential(nn.Linear(width + 12 + 16, 64), nn.ReLU(), nn.Linear(64, width))

    def encode_users(self, users, hours, histories):
        history_sum = self.item_id(histories + 1).sum(dim=1)
        history_mean = history_sum / (histories >= 0).sum(dim=1, keepdim=True).clamp_min(1)
        user_embedding = self.user_id(users + 1)
        if not self.use_user_id:
            user_embedding = torch.zeros_like(user_embedding)
        vectors = self.user_net(torch.cat((user_embedding, history_mean, self.hour(hours)), dim=1))
        return F.normalize(vectors, dim=1)

    def encode_items(self, items):
        vectors = self.item_net(torch.cat((
            self.item_id(items + 1),
            self.category(self.categories[items]),
            self.text_projection(self.text[items]),
        ), dim=1))
        return F.normalize(vectors, dim=1)


def fit_tower(data, train, *, seed=42, epochs=10, correction=False, use_user_id=True):
    torch.manual_seed(seed)
    expanded = [(*row[:4], target, i) for i, row in enumerate(train) for target in row[4]]
    target = torch.tensor([row[4] for row in expanded], dtype=torch.long)
    counts = torch.bincount(target, minlength=data.catalog.size)
    log_q = ((counts + 1) / (len(target) + data.catalog.size)).log()
    known_users = torch.zeros(data.users, dtype=torch.bool)
    known_users[[row[0] for row in train]] = True
    model = TwoTower(data, use_user_id=use_user_id)
    with torch.no_grad():
        model.user_id.weight[1:][~known_users] = 0
        model.item_id.weight[1:][counts == 0] = 0
    loader = DataLoader(
        TensorDataset(*tensor_contexts(expanded), target, torch.tensor([row[5] for row in expanded])), batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses = []
    for _ in range(epochs):
        total_loss = 0.0
        for user, hour, history, items, request_ids in loader:
            unique_items, labels = items.unique(return_inverse=True)
            query = model.encode_users(user, hour, history)
            candidates = model.encode_items(unique_items)
            scores = query @ candidates.T / 0.1
            if correction:
                # Columns are deduplicated, so correct by the probability of inclusion.
                q = log_q[unique_items].exp().to(torch.float64)
                inclusion = -torch.expm1((len(items) - 1) * torch.log1p(-q))
                scores = scores - inclusion.clamp_min(1e-12).log().to(scores.dtype)
            # Other known targets for the same request are positives, not negatives.
            item_columns = {item: col for col, item in enumerate(unique_items.tolist())}
            co_positives = [
                (batch_row, item_columns[item])
                for batch_row, (request_id, target_item) in enumerate(zip(request_ids.tolist(), items.tolist(), strict=True))
                for item in train[request_id][4]
                if item != target_item and item in item_columns
            ]
            if co_positives:
                rows, cols = zip(*co_positives)
                scores[rows, cols] = -torch.inf
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(items)
        losses.append(total_loss / len(target))
    return model.eval(), losses


@torch.inference_mode()
def vectors(model, contexts):
    inputs = tensor_contexts(contexts)
    items = model.encode_items(torch.arange(len(model.categories))).cpu().numpy()
    users = np.concatenate([
        model.encode_users(*(tensor[start:start + 512] for tensor in inputs)).cpu().numpy()
        for start in range(0, len(contexts), 512)
    ])
    return users, items
