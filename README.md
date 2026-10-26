# cmip6_input

CMIP6 の 42 モデル (1850〜2014 年、165 年 × 12 か月) の前処理済みデータを、CNN の入力 (predictors) と出力 (降水量 pr の連続値・クラス値) に変換するスクリプト群です。
パスはすべて解析サーバ上の絶対パス (`/work/kajiyama/...`) を前提にしています。

## ファイル構成と実行順

| 順 | スクリプト | 入力 | 出力 |
|---|---|---|---|
| 1 | `pickle_to_npy.py` | `/work/kajiyama/preprocessed/cmip6/<model>/<var>_<model>.pickle` | `/work/kajiyama/cnn/input/<var>/<var>_{raw,clim,variance,anom,std,coarse,...}.npy` (モデル軸を結合) |
| 2 | `mk_predictors.py` | `/work/kajiyama/cnn/input/{mrso,snc,tos,tsl}/<var>_<key>.npy` | `/work/kajiyama/cnn/input/predictors/predictors_<key>_<月>_<組合せ>.npy` (4 変数の 15 通りの組合せ, m/s/o/t) |
| 3 | `mk_pr.py` | `/work/kajiyama/cnn/input/pr/main/pr_<form>.npy` | `/work/kajiyama/cnn/input/pr/pr_<解像度>_<form>_<月>_<領域>.npy` (1x1 と 5x5 の両方) |
| 4 | `mk_class.py` | `/work/kajiyama/cnn/input/pr/continuous/<領域>/<解像度>/pr_<key>_std_MJJASO_<領域>.npy` | `/work/kajiyama/cnn/input/pr/class/<領域>/<EFD|EWD>/..._<EFD|EWD>_<クラス数>.npy` |
| 5 | `mk_test.py` | predictors と pr の連続値 | `/work/kajiyama/cnn/test/{train,val}/..._1973-2014.npy` (各モデル 1 年をテストに使う分割) |

* `discrete.py` は離散化の共通モジュールです (`efd_bnd`, `ewd_bnd`, `classify`, `ewd_report`)。`mk_class.py` から import されます。旧来の `EFD()` / `EWD()` 関数もそのまま使えます。
* EFD は等頻度分割、EWD は 0 を中心とした等幅分割です。

## 実行方法

```bash
python pickle_to_npy.py
python mk_predictors.py
python mk_pr.py
python mk_class.py      # SAVE_FLAG (ファイル先頭) が True のとき保存
python mk_test.py
```

主な設定はそれぞれのスクリプト先頭の定数で変更します。

* `mk_pr.py` : `SPACE_LST` (`one`, `thailand`, `monsoon`, `world`)、`LEAD_LST`、解像度ごとの設定 `RES_CONFIG`
* `mk_class.py` : `SAVE_FLAG`、`exec_resolution_discrete_class_space()` 内のクラス数・手法・解像度・領域のリスト、格子数 `GRID`
* `mk_test.py` : `main()` 内の入力ファイル名

## 依存パッケージ

numpy, matplotlib, cartopy (`mk_class.py` の描画のみ)
