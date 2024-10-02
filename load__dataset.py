import numpy as np
import pandas as pd
def load_UNSW_Flow():
    train = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_train_use.csv")
    train = train[train['binary_label_attack'] == 0].drop(columns=['timestamp', 'label_background','label_exploits','label_fuzzers','label_reconnaissance','label_dos','label_analysis','label_backdoor','label_shellcode','label_worms','label_other','binary_label_normal','binary_label_attack'], axis=1).values
    x_train = torch.from_numpy(train).float()
    y_train = torch.zeros(len(x_train))

    test = pd.read_csv("/root/bishe/dataset/UNSW/UNSW_Flow_test_use.csv")
    y_test = torch.from_numpy(test['binary_label_attack'].apply(lambda x: 0 if x == 0 else 1).values)
    x_test = test.drop(columns=['timestamp', 'label_background','label_exploits','label_fuzzers','label_reconnaissance','label_dos','label_analysis','label_backdoor','label_shellcode','label_worms','label_other','binary_label_normal','binary_label_attack'], axis=1).values #an m-by-n dataset with m observations
    x_test = torch.from_numpy(x_test).float()
    print(x_train)
    print(x_train.shape)
    print(y_train)
    print(x_test)
    print(x_test.shape)
    print(y_test)
    return (x_train, y_train), (x_test, y_test)