# lightning-hydra

# trial task
* 6class分類
    * BasicSystemを構築（ひとまずRegを継承して作ってあとで入れ替える）
* Smooth L1（その場合、valにRMSEも加える）
* Tile aug（サブする、なぜかLB0.84...）
    * おそらくscaling factorが大きすぎたのが原因
    * これくらいでも良いかも、なんなら平均は1じゃなくてもっと小さくても良いかも
    * np.clip(np.random.normal(loc=1, scale=0.5, size=1), 0.5,3.5)
* pool4x4 （様子見、良くも悪くもない）
* GeM
* Lazy Accuracyの実装

# improve task
* korniaによるDA高速化

# 完了タスク
* 保存するckptを20に