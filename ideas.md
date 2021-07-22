# 2021/07/09

- ルールと課題を確認する
    - 提出するもの
        - CSVだけ出せばよい。
        - `target` という名前のカラムだけを含む。
        - 評価指標はRMSE。
        - Ground truth は `[0, 1, 2, 3]` だが、`0.2` みたいな小数値を出力してもOK。
    - テストで使えるもの
        - 入力は画像のみ。画像から年代を推定する。
    - トレーニングで使えるもの
        - photos: 画像
        - materials: 作品の材料
            - 1つの絵画は複数の材料で構成される。
        - techniques: 技法
            - 1つの絵画に複数の技法が使われることがある。
- モデルv1.0
    - 画像からラベル (float値1つ) を出力する
    - 特徴抽出(CNN) + 回帰(Linear)
        - CNNを線形回帰モデルにするには、CNNの後ろにLinear層を追加すればよい。
        - Dropoutを入れておく。
        - 学習率はとりあえず手動で決める。
        - オプティマイザはSGD、スケジューラはまだ設定しない。
        - lossはfloat値に対するRMSEを使う。
            - 単位が「年」になるので、何年くらいずれているかがわかる。
            - 学習が正しく進むと、lossが 100 (100年) を下回る。
        - バリデーションセットに対しては、カテゴリ値に対するRMSEも出しておく。
            - リーダーボードの値はこちらが使われる。
            - 学習が正しく進むと、およそ1.0を下回る。
    - データ拡張
        - とりあえず回転、反転くらいは入れておく。
        - Min-max正規化を適用しておく。
    - データ分割
        - 提出を早めに試したいので、とりあえずrandom_splitを使う。
- フレームワーク
    - PyTorchを採用
        - pytorch-lightning を初めて試してみる。
        - 使い心地を確認したい。
    - pytorch-lightning
        - ログ
            - ロガーを差し替え可能らしい。
            - 使い方が頻繁に変わるのか、ブログ記事などがあてにならないことが多い。
                - 公式サイトを信じる。特に全文検索ができるpdf版ドキュメントが便利。

# 2021/07/10

- モデルv1.1
    - Warmupを試す
        - OneCycleLR
        - pytorch-lightning の初期設定では、ステップ毎に `optimizer.step()` が呼ばれない。
            - [configure_optimizers()](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers) に設定方法の記載がある。
        - 学習率のモニタリング
            - [LearningRateMonitor](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.LearningRateMonitor.html) を使って、ステップ毎やエポック毎に記録が可能。
    - 交差検証を導入
        - とりあえず5分割する
        - 不具合がないかの確認のため、提出時に `train.csv` に対しても予測し、参考スコアを出すようにする
- submission v1
    - 推論結果
        - train: 0.80くらい
        - public: 0.7683

# 2021/07/11

- materials.csv を年代ごとの採用率に着目して眺めてみる
    - 絵画に採用される材料の種類は徐々に増えていく。歴史とともに新しい材料が発明されていく。
    - ink の採用率は初期のほうが高い。
- テーブルデータをどう使うか?
    - バイナリラベルに変換してあげて、線形回帰の入力に加える
        - テストデータにはバイナリラベルがないので、準備が大変そう
    - 描かれた年だけでなく、materialsも予測するようなモデルを作る
        - 線形回帰の出力を増やしてあげるだけでよさそう。
        - 最適化の過程ではバイナリラベルとのBinaryCrossEntropyをとればよい。
        - 推論時は年の出力だけ見ればよい。
        - ある程度サンプル数が必要。materials のうち、多く採用されているものを使いたい。
- materials.csv を使わずに、質感などをモデルに反映できないか?
    - 描かれているものは関係なく、絵画の質感だけを見てほしい
    - 絵をばらしてランダムに組み替えればよいのでは?
        - [Concat tile pooling](https://www.kaggle.com/razamh/panda-concat-tile-pooling-starter-0-79-lb)
        - [Random Grid Shuffle](https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RandomGridShuffle)


# 2021/07/12

- 採用率が高く、年代を経ると増減する材料を予測してみる
    - ink, pencil, watercolor (paint)

# 2021/07/14

- version_19 (削除済み)
- Schedulerを停止し、固定学習率でAdamをひたすら回してみる
    - epoch<100 くらいで停滞する。val_lossは約85。
    - 一方、train_lossは落ち続ける。学習率が少し大きすぎる可能性がある。
- version_20
    - Schedulerを指数減少に切り替え (epoch**0.95)、ひたすら回してみる
    - Warmupと同じ降下のしかたで、lossの振幅を大きく抑えることができた。
    - 学習率を下げることに意味はありそう。
    - Warmupと見比べてみる
        - Warmupで非常に小さい学習率になったあたりから、一気にlossが落ち始めている。
        - 落とし方のコツがありそう
    - いくつか試していると、降下が大きくなる学習率がいくつかあり、それをいかに早く見つけ出すかがポイントになりそう
        - 5e-4, 1e-4、あとどこ?
        
- version_25 (削除済み)
    - 階段状になることが分かってきたので、初期学習率だけ決めて `ReduceLROnPlateau` を使ったほうが早そう
        - あたりがついた後に StepLR へ置き換えたい
    - 初期学習率 5e-4 で階段状に落とす
        - factor は調整が必要そう。デフォルト値は大きすぎて効果のある学習率 (1e-4) を飛ばしてしまう。
- version_26-30
    -  `ReduceLROnPlateau` で0.25倍するのを繰り返してみる
    - 精度はまあまあだが、適切な学習率はむしろわかりにくくなっている。

# 2021/07/15

- findLRで探索するほうが早く終わるのでは?

# 2021/07/16

- 実験環境を整える
    - どのモデルからトレーニングしたかを記録する

# 2021/07/19

- ソースコードのブラッシュアップ
    - モデル定義をあるべき方法に直す
        - pytorchモデルはFC層が末端についており、`model.fc` でアクセスできる。これを上書きして、任意のLinearレイヤに差し替えることができる。
        - timmモデルはFC層に `model.classifier` でアクセスできる。これも上書き可能。
    - パラメータ管理をちゃんとやる
        - 探せばPyTorch-Lightningのドキュメント、またはDiscussionにちゃんと書いてある
- Optimizer, Schedulerの実験結果について
    - warmupはあったほうがよい。lossを一気に落とせる。
        - 1cycle policyというらしい。
            - https://fastai1.fast.ai/callbacks.one_cycle.html
            - cos, linear どちらでも落ちる。
    - warmupの代わりにexponentialスケジューラを使ったモデルを再トレーニングし続けることでも、同様の効果が得られる。
    - いずれにせよ、スクラッチ学習をひたすらやるより、モデルを徐々に「温める」のがよさそう。
        - augmentationを強くするのは「温まってから」で良いのでは?
            - EfficientNetV2とか確かそんな感じ
- モデルの分析
    - 作成したモデルのFC層を眺めてみる
        - validationデータとTensorboardを用いて、PCAやt-SNEを適用してみる。

# 2021/07/22

- 学習曲線をひたすら眺める
    - n_featuresを増やすとtrain_lossの初動は大きく落ちる
        - 落とさなくてもval_lossは落ちるし、乖離は残る。初動を気にしなくてもよいのでは?
    - ExponentialLRのパラメータを色々変えてみる
        - 滑らかにはなるが、80台を下回るのが難しい。一定の学習率で固定したほうがよさそうな感じがしてきた。
    - StepLRに切り替える
        - warmupなくてもある程度下がることは確認済み
        - 70/70/60, gamma=0.25
            - 終盤はもうちょっと長めに見てよさそう
    - Augmentation
        - 明るさやコントラストを変えてみる
            - むしろ悪化した。val_lossが90くらいで止まってしまう。
- TTA入れるか、入れないか
    - submission v2候補
    - TTAなし
        - train: 0.7987
    - TTAあり (n_tta=4, seed=2021)
        - train: 0.7813
