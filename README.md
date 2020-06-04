# lightning-hydra

# trial task
* hydraのconfig整備、cyclicLRの部分とか特に
    * 引数でconfigファイルを指定できるように
    * 差分configファイルを適用できるように

* 6class分類（様子見、なんかaccだけ異様に良い）
* Smooth L1（様子見）
* Tile aug（サブする、なぜかLB0.84...）
    * おそらくscaling factorが大きすぎたのが原因
    * これくらいでも良いかも、なんなら平均は1じゃなくてもっと小さくても良いかも
    * np.clip(np.random.normal(loc=1, scale=0.5, size=1), 0.5,3.5)
* pool4x4 （様子見、良くも悪くもない、サブするほどでも無い感）
* GeM

# improve task
* korniaによるDA高速化

# 完了タスク
* 保存するckptを20に
* Lazy Accuracyの実装：monitored kappaで解決
* 学習の高速化（次の日にはサブできるくらいに）：restart + cyclicLRで解決