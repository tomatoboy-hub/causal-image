import sys
sys.path.append("../../")
sys.path.append("../../../")

from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
import cv2

def preprocessing(input_df):
    """
    欠損値や文字の処理を行う
    """
    def price2yen(input_df):
        output = input_df.copy()
        output = output.dropna(subset=["actual_price"])
        output["actual_price"] = output["actual_price"].str.strip("₹")
        output["actual_price"] = output["actual_price"].str.replace(",","").astype(float)
        output["actual_price_yen"] = output["actual_price"] * 110
        return output
    
    def no_of_rate(input_df):
        output = input_df.copy()
        output = output.dropna(subset=["no_of_ratings"])
        #文字情報が含まれている場合、エラーが発生する可能性があるため、エラーハンドリングを追加
        output["no_of_ratings"] = pd.to_numeric(output["no_of_ratings"].str.replace(",",""), errors='coerce').fillna(0).astype(int)
        return output
    
    def img_path(input_df):
        output = input_df.copy()
        output = output.dropna(subset=["img_path"])
        return output
    
    #output = price2yen(input_df)
    #output = no_of_rate(output)
    output = img_path(input_df)
    return output

def filter_outlier(output_df):
    # 四分位範囲（IQR）の計算
    Q1 = output_df["actual_price_yen"].quantile(0.25)
    Q3 = output_df["actual_price_yen"].quantile(0.75)
    IQR = Q3 - Q1

    # 外れ値の閾値を設定
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 外れ値を除去
    filtered_df = output_df[(output_df["actual_price_yen"] >= lower_bound) & (output_df["actual_price_yen"] <= upper_bound)]
    return filtered_df

def make_confounder(filtered_df):
    def calculate_sharpness(img_path):
        # 画像をグレースケールで読み込む
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # ラプラシアンフィルタを適用して勾配を計算
        laplacian = cv2.Laplacian(img, cv2.CV_64F)

        # ラプラシアンの分散をシャープネスとして使用
        sharpness = np.var(laplacian)

        return sharpness
    filtered_df["sharpness"] = filtered_df["img_path"].apply(calculate_sharpness)
    # "actual_price_yen"の平均を計算
    mean_edge = filtered_df["sharpness"].mean()
    print(mean_edge)
    # "price_ave"列を追加
    filtered_df["sharpness_ave"] = filtered_df["sharpness"].apply(lambda x: 1 if x > mean_edge else 0)
    return filtered_df

def make_confounder_include_text(filtered_df):
    import easyocr
    def contains_text_easyocr(img_path, languages=['en']):
        # EasyOCRリーダーを初期化（言語を指定）
        reader = easyocr.Reader(languages)

        # OCRでテキストを抽出
        results = reader.readtext(img_path)
        print(results)
        # 結果が空でないかを判定
        if results:
            # 抽出されたテキストをリストにまとめる
            extracted_text = [result[1] for result in results]  # result[1] は抽出された文字列
            return 1
        else:
            return 0
    filtered_df["contains_text"] = filtered_df["img_path"].apply(contains_text_easyocr)
    return filtered_df

def make_confounder_tesseract_text(filtered_df):
    import pytesseract
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
    filtered_df["contains_text"] = filtered_df["img_path"].apply(contains_text)
    return filtered_df

def make_treatment(df):
    def is_dark_or_light(image_path, threshold=160):
        # 画像を読み込んでRGBに変換
        img = Image.open(image_path).convert('RGB')
        # 画像をNumPy配列に変換
        img_np = np.array(img)
        # 輝度の計算 (R, G, B の加重平均)
        brightness = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
        # 画像全体の平均輝度を計算
        avg_brightness = np.mean(brightness)
        print(f"平均輝度: {avg_brightness}")
        # 平均輝度が閾値より低ければ「黒っぽい」、高ければ「白っぽい」
        if avg_brightness < threshold:
            print("画像は黒っぽいです。")
            return "dark"
        else:
            print("画像は白っぽいです。")
            return "light"
    
    df["light_or_dark"] = df["img_path"].apply(is_dark_or_light)    
    df["light_or_dark"] = df["light_or_dark"].apply(lambda x : 1 if x == "light" else 0)
    randoms = np.random.uniform(0, 1, len(df['light_or_dark']))
    accuracy = -1
    if isinstance(accuracy, tuple): # TODO this is a hack
        pThatGivenT = accuracy
    elif accuracy > 0:
        pThatGivenT = [1 - accuracy, accuracy]
    else:
        pThatGivenT = [0.2, 0.8]
    mask = np.array([pThatGivenT[ti] for ti in df['light_or_dark']])
    df['T_proxy'] = (randoms < mask).astype(int)
    return df

if __name__ == "__main__":
    csv_path = "/root/graduation_thetis/causal-bert-pytorch/input/backlog/csv/All Appliances_preprocess.csv"
    df = pd.read_csv(csv_path)
    df = preprocessing(df)
    #df = filter_outlier(df)
    #df = make_confounder(df)
    #df = make_confounder_include_text(df)
    df = make_confounder_tesseract_text(df)
    df = make_treatment(df)
    df.to_csv("/root/graduation_thetis/causal-bert-pytorch/input/Appliances_preprocess_1125.csv",index = None)
