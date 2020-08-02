# lightning-hydra
## Idea
* co teaching
* O2U-Net
* Smooth L1 Loss
* 強いモデルアーキテクチャ
* あとは学習中に動的にラベルを更新していく系
* (noise layer)
* cleanlab

## input data
* タイル化
    * window size固定128 x 16タイル
    * window size auto x　16タイル（layer0より）これlayer1でもっとscaleさせてもう一度試す。
* data augmentation
    * v1とv2のみしか試していない。v2の方が理にかなったaugmentationしているはずだけどv1の方が微妙にいい？？

## model
* モデルアーキテクチャ
    * se resnet50のみ

* 出力
    * regression
    * classification
    
* カスタム
    * avg
    * avg + max

## Optimize method
* loss
    * RMSE Loss
    * Cross Entropy


# trial task
* Smooth L1（様子見）
* Tile aug（サブする、なぜかLB0.84...）
    * おそらくscaling factorが大きすぎたのが原因
    * これくらいでも良いかも、なんなら平均は1じゃなくてもっと小さくても良いかも
    * np.clip(np.random.normal(loc=1, scale=0.5, size=1), 0.5,3.5)
    * ここまで全てopenslideのバグ！！！

* pool4x4 （様子見、良くも悪くもない、サブするほどでも無い感）
* GeM

# improve task
* korniaによるDA高速化

# 完了タスク
* 保存するckptを20に
* Lazy Accuracyの実装：monitored kappaで解決
* 学習の高速化（次の日にはサブできるくらいに）：restart + cyclicLRで解決
* hydraのconfig整備、cyclicLRの部分とか特に
    * 引数でconfigファイルを指定できるように
    * 差分configファイルを適用できるように
* 6class分類（様子見、なんかaccだけ異様に良い、サブしたら0.86、0.87のregressionと混ぜると0.88になった）
* monitored qwkの最後の正規化を修正（class-1に）
* base.yamlとoverride.yamlを分ける作業
