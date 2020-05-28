## Setting parameters

class config:
    #preprocess
    max_len = 64
    batch_size = 64
    train_path = "./data/ratings_train.txt"
    test_path = "./data/ratings_test.txt"
    kaggle_path = "./data/ko_data.csv"

    #train
    warmup_ratio = 0.1
    num_epochs = 50
    max_grad_norm = 1
    log_interval = 200
    learning_rate =  5e-5
