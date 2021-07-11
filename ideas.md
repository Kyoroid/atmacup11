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