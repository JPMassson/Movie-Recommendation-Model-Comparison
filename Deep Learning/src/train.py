import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
import numpy as np
from torch.utils.data import DataLoader

class MovieDataset:
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, item):
        user = self.users[item]
        movie = self.movies[item]
        rating = self.ratings[item]

        return {
            "users": torch.tensor(user, dtype=torch.long),
            "movies": torch.tensor(movie, dtype=torch.long),
            "ratings": torch.tensor(rating, dtype=torch.float)
        }


class RecSysModel(nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, 64)
        self.movie_embed = nn.Embedding(num_movies, 64)
        self.out = nn.Linear(128, 1)

    def forward(self, users, movies, ratings=None):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        output = torch.cat([user_embeds, movie_embeds], dim=1)
        output = self.out(output)

        loss = nn.MSELoss()(output, ratings.view(-1, 1))
        return output, loss


def train():
    df = pd.read_csv("../input/train_v2.csv")
    lbl_user = preprocessing.LabelEncoder()
    lbl_movie = preprocessing.LabelEncoder()

    df.user = lbl_user.fit_transform(df.user.values)
    df.movie = lbl_movie.fit_transform(df.movie.values)

    df_train, df_valid = model_selection.train_test_split(df, test_size=0.1, random_state=42, stratify=df.rating.values)

    train_dataset = MovieDataset(users=df_train.user.values, movies=df_train.movie.values, ratings=df_train.rating.values)
    valid_dataset = MovieDataset(users=df_valid.user.values, movies=df_valid.movie.values, ratings=df_valid.rating.values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RecSysModel(num_users=len(lbl_user.classes_), num_movies=len(lbl_movie.classes_))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    for epoch in range(5):
        model.train()
        for batch in train_loader:
            users = batch['users'].to(device)
            movies = batch['movies'].to(device)
            ratings = batch['ratings'].to(device)

            optimizer.zero_grad()

            output, loss = model(users, movies, ratings)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                users = batch['users'].to(device)
                movies = batch['movies'].to(device)
                ratings = batch['ratings'].to(device)

                output, loss = model(users, movies, ratings)
                val_loss += loss.item()

        val_loss /= len(valid_loader)

        print(f"Epoch: {epoch+1}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    train()
