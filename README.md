# atmaCup11

コンテストtop  
https://www.guruguru.science/competitions/17/

開催期間  
2021/07/09 18:00 〜 2021/07/22 18:00

## 順位

private: 67th (0.7407)

## 解法

### 解法1 (2021/07/11)

コード  
https://github.com/Kyoroid/atmacup11/tree/v1

提出物 & ログ  
https://github.com/Kyoroid/atmacup11/tree/v1/artifacts/submission_v1

- 5-fold CV
    - GroupKFold
- Architecture
    - ResNet18
    - regression
- Augmentation
    - train
        - HorizontalFlip, Pad, RandomCrop, RandomBrightness
    - val/test
        - Pad, CenterCrop
- Optimizer
    - Adam (init_lr=5e-4)
    - Cosine annealing (warmup=10%)
- Training
    - 100epoch
    - Batch size: 32
- Submission
    - submission.csv は各foldの平均から作成 (提出1)


### 解法2 (2021/07/22、最終提出)

コード  
https://github.com/Kyoroid/atmacup11/tree/v2

提出物 & ログ  
https://github.com/Kyoroid/atmacup11/tree/v2/artifacts/submission_v2

- 5-fold CV
    - GroupKFold
- Architecture
    - ResNet18
    - regression
- Augmentation
    - train
        - Flip, Pad, RandomCrop
    - val
        - Pad, CenterCrop
    - test/submit (TTA)
        - Flip, Pad, RandomCrop
- Optimizer
    - Adam (init_lr=5e-4)
    - StepLR (70epoch毎に1/4)
- Training
    - 200epoch
    - Batch size: 64
- Submission
    - TTA * 4
    - submission.csv は各foldの平均から作成 (提出2、リーダーボードのスコアはこれ)
    - targetの値を丸めたものを作成 (提出3)

## Colabでの実行例

交差検証を実施する

[atmacup11.ipynb](atmacup11.ipynb)