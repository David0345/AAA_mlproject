import gdown
import os


def download_data():
    """
    Создает директорию image_embeddings и скачивает туда 3 файла:
        test_embeddings.parquet, shape=(N, 513),
        train_embeddings.parquet, shape=(N, 513),
        efficientnet_b6_weights.pth, weights for model
    """

    save_dir = '../image_embeddings'
    os.makedirs(save_dir, exist_ok=True)

    files = {
        'test_embeddings.parquet':
            'https://drive.google.com/file/d/1zTpekL83Q4L_-vFuYw0bf_BgvXjGZ5yd/view?usp=sharing',
        'train_embeddings.parquet':
            'https://drive.google.com/file/d/1BSDjA_ZXMQPDB-BoKxKzDjPCGeqAnFg4/view?usp=sharing',
        'efficientnet_b6_weights.pth':
            'https://drive.google.com/file/d/1ZEYy__gg_QQHoTvTwpa5cWUX566ExrD-/view?usp=sharing'
    }

    for filename, file_url in files.items():
        filepath = os.path.join(save_dir, filename)

        if not os.path.exists(filepath):
            print(f"Скачиваю {filename}...")
            gdown.download(
                url=file_url,
                output=filepath,
                quiet=False,
                fuzzy=True
            )

            print(f'{filename} скачан')
        else:
            print(f'{filename} уже существует')


if __name__ == '__main__':
    download_data()
