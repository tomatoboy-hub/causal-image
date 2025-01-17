import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
def gather_image_paths(folder_path, extensions=(".jpg", ".jpeg", ".png", ".gif")):
    """
    指定フォルダ内の画像ファイルパスをリスト化して返す。
    デフォルトでは jpg, jpeg, png, gif を対象とする。
    """
    file_paths = []
    for filename in os.listdir(folder_path):
        # 拡張子の小文字変換で比較し、画像ファイルのみを対象にする
        if filename.lower().endswith(extensions):
            # 絶対パス or 相対パスを作成
            file_paths.append(os.path.join(folder_path, filename))
    return file_paths

def calculate_sharpness(img_path):
    """
    画像をグレースケールで読み込み、ラプラシアンの分散からシャープネスを計算。
    値が大きいほどエッジが多い(=シャープ)。
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan  # 読み込み失敗時は NaN などを返す
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    return np.var(laplacian)

def calculate_brightness(img_path):
    """
    画像をRGBで読み込み、輝度(0.299*R + 0.587*G + 0.114*B)の平均を返す。
    値が大きいほど明るい画像。
    """
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        # 輝度(Y) = 0.299R + 0.587G + 0.114B
        brightness = 0.299*img_np[:, :, 0] + 0.587*img_np[:, :, 1] + 0.114*img_np[:, :, 2]
        return np.mean(brightness)
    except:
        return np.nan
    
def make_confounder_tesseract_text(filtered_df):
    def contains_text(img_path):
        # 画像を読み込む
        img = cv2.imread(img_path)

        # グレースケールに変換（OCRに適しているため）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OCRでテキストを抽出
        extracted_text = pytesseract.image_to_string(gray)
        # 抽出されたテキストが空でないかを判定
        if extracted_text.strip():
            return 1
        else:
            return 0  # 文字が含まれていない
    filtered_df["contains_text"] = filtered_df["file_path"].apply(contains_text)
    return filtered_df


def main():
    # 1) allimages フォルダから画像ファイルを収集
    folder_path = "/root/graduation_thetis/causal-bert-pytorch/input/shoclo_imgs"  # 画像があるフォルダ
    image_paths = gather_image_paths(folder_path)

    # 2) DataFrame を作成 (file_path 列のみ)
    df = pd.DataFrame({"file_path": image_paths})

    # 3) シャープネスを計算し、中央値より大きいかどうかで二値化
    df["sharpness"] = df["file_path"].apply(calculate_sharpness)
    median_sharpness = df["sharpness"].median()  # シャープネスの中央値
    df["sharpness_ave"] = df["sharpness"].apply(lambda x: 1 if x > median_sharpness else 0)

    # 4) 明るさ (平均輝度) を計算し、中央値より大きいかどうかで二値化
    df["avg_brightness"] = df["file_path"].apply(calculate_brightness)
    median_brightness = df["avg_brightness"].median()  # 平均輝度の中央値
    df["light_or_dark"] = df["avg_brightness"].apply(lambda x: 1 if x >= median_brightness else 0)

    df = make_confounder_tesseract_text(df)


    # 5) 出力用に必要な3列だけに絞る
    df = df[["file_path", "sharpness_ave", "light_or_dark","contains_text"]]

    # 6) CSVを保存 (今回は同ディレクトリに output.csv を作成する例)
    output_csv_path = "shoclo_output.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"新規CSVを作成しました: {output_csv_path}")

if __name__ == "__main__":
    main()
