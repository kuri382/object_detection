"""detect objects
輪郭抽出および抽出した画像を分割して保存する
"""

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_contour(file, srcmin, srcmax, areamin, areamax):
    """detect_contour
    画像ファイルを読み込み，変換し輪郭抽出を行う
    args
        file:ファイル名
        srcmin:閾値
        srcmax:閾値以上の値を持つ画素に割り当てられる値
        areamin:判定する輪郭サイズの最小値（これ以上はノイズとする）
        areamax:判定する輪郭サイズの最大値（これ以上は判別しない）
    """
    # 画像を読込
    org_src = cv2.imread('data/'+file+'.jpg', cv2.IMREAD_COLOR)
    src = np.copy(org_src)

    # グレースケール画像へ変換
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("split_image/{}/gray.jpg".format(file), gray)

    # グレースケール画像2値化
    _, bw = cv2.threshold(
        gray, srcmin, srcmax, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("split_image/{}/binary.jpg".format(file), bw)

    # 輪郭を抽出
    #   contours : [領域][Point No][0][x=0, y=1]
    #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
    #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
    contours, _ = cv2.findContours(
        bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 判別されたオブジェクトの数を初期化
    brain_n = 0
    # 各輪郭に対する処理
    for contour in contours:
        # 輪郭の領域を計算
        area = cv2.contourArea(contour)

        # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
        if area < areamax or areamin < area:
            continue

        # 外接矩形
        if len(contour) > 0:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 5)
            # 外接矩形毎に画像を保存
            cv2.imwrite(
                "split_image/{}/split{:0>3}.jpg".format(file, brain_n), org_src[y:y + h, x:x + w])
            # オブジェクトの数を更新
            brain_n += 1

    # 外接矩形された画像を表示
    #cv2.imshow('output', src)
    #plt.imshow(src)
    #test = input()
    #plt.close()
    # 外接矩形された画像をsave
    cv2.imwrite("split_image/{}/all_contours.jpg".format(file), src)


if __name__ == '__main__':
    # 画像処理条件
    SRC_MIN = 100 #二値化の閾値
    SRC_MAX = 255 #二値処理後の最大値（黒）
    AREA_MIN = 1e7 #このサイズ以下のものをオブジェクトと判定
    AREA_MAX = 1e3 #このサイズ以上のものをオブジェクトと判定

    #'data'ディレクトリ内の全画像ファイル名を取得し，一件ずつFILE_NAMEに代入
    for FILE_NAME in [os.path.splitext(x)[0] for x in sorted(os.listdir('data'))]:
        # 元画像ファイルに応じたsplit_imageディレクトリを作成
        if not os.path.isdir('split_image/'+FILE_NAME):
            os.makedirs('split_image/'+FILE_NAME)

        # 実行処理
        print("Processing {}.jpg...".format(FILE_NAME))
        detect_contour(FILE_NAME, SRC_MIN, SRC_MAX, AREA_MIN, AREA_MAX)
