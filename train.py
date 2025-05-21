from dataset import ShakespeareDataset
from model import TokenEmbedding




def train():
    dataset = ShakespeareDataset('train', 1024)
    token_embedding = TokenEmbedding(300000, dataset.tokenizer.pad_token_id)


if __name__ == '__main__':
    train()
