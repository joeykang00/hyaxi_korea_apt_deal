# pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn pillow

import pandas as pd
import os
import zipfile
import sys
import joblib
import platform
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from PIL import Image

TRAINED_DATA_DIR = 'trained_data'
MODEL_FILE_PATH = os.path.join(TRAINED_DATA_DIR, 'xgb_apartment_model.joblib')
PRELOAD_FILE_PATH = os.path.join(TRAINED_DATA_DIR, 'preload_xgb_data.csv')
ENCODERS_FILE_PATH = os.path.join(TRAINED_DATA_DIR, 'label_encoders.joblib')
PLOT_OUTPUT_DIR = 'results'  # 그래프 및 추천 결과 저장 폴더

SHOULD_RETRAIN = False
MIN_TRANSACTION_COUNT = 100  # 최소 거래 횟수 필터링 기준
cat_features = ['시도명', '시군구명', '법정동', '아파트']

# --------------

def set_font():
    os_name = platform.system()
    font_family = None

    if os_name == 'Windows':
        font_list = ['Malgun Gothic', 'Dotum', 'Gulim']
        for font in font_list:
            try:
                if fm.findfont(font, fallback_to_default=False):
                    font_family = font
                    break
            except:
                continue

        if font_family:
            plt.rc('font', family=font_family)
        else:
            print("Warning: Could not find a suitable Hangul font (Malgun Gothic, Dotum, Gulim) on this Windows system. Characters may appear broken.")

    elif os_name == 'Darwin':  # macOS
        font_list = ['Apple SD Gothic Neo', 'AppleGothic']
        for font in font_list:
            try:
                if fm.findfont(font, fallback_to_default=False):
                    font_family = font
                    break
            except:
                continue

        if font_family:
            plt.rc('font', family=font_family)
        else:
            print("Warning: Could not find a suitable Hangul font (Apple SD Gothic Neo, AppleGothic) on this macOS system. Characters may appear broken.")

    elif os_name == 'Linux':
        try:
            if fm.findfont('NanumGothic', fallback_to_default=False):
                plt.rc('font', family='NanumGothic')
            else:
                print("Warning: 'NanumGothic' not found on this Linux system. Please install it. Characters may appear broken.")
        except Exception as e:
            print(f"Warning: An error occurred while setting 'NanumGothic'. Details: {e}")

    else:
        print("Warning: Could not determine the OS to set a proper Hangul font. Characters may appear broken.")

    # 공통: 음수 부호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False


# **********************************************

def format_manwon(amount):
    # amount가 넘파이 float 타입일 수 있으므로 int로 변환
    if pd.isna(amount):
        return "N/A"
    return f"{int(amount):,.0f} 만 원"


def prepare_csv_from_zip(data_dir, csv_filename, zip_filename):

    csv_path = os.path.join(data_dir, csv_filename)
    zip_path = os.path.join(data_dir, zip_filename)

    # 분할 압축 파일의 첫 번째 파일 경로 (예: KoreaApartDeal_PreProcessed.zip.001)
    split_zip_start = zip_path + ".001"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(csv_path):
        print(f"'{csv_path}' is not exist. Checking Zip file.")

        # 1. 단일 Zip 파일이 있는 경우
        if os.path.exists(zip_path):
            print(f"'{zip_path}' is found. Unzip...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"Completed Unzip. '{csv_path}' will be used.")
            except Exception as e:
                print(f"Unzip error: {e}")
                sys.exit(1)

        # 2. 단일 Zip은 없고, 분할 압축 파일(.001)이 있는 경우
        elif os.path.exists(split_zip_start):
            print(f"Split zip file found ('{os.path.basename(split_zip_start)}'). Merging and Unzipping...")

            # 병합을 위한 임시 파일
            temp_merged_zip = os.path.join(data_dir, "temp_merged_archive.zip")

            try:
                # (1) 분할 파일 병합
                with open(temp_merged_zip, 'wb') as merged_f:
                    part_num = 1
                    while True:
                        part_file = f"{zip_path}.{part_num:03d}"
                        if not os.path.exists(part_file):
                            break

                        # print(f"Merging {os.path.basename(part_file)}...")
                        with open(part_file, 'rb') as part_f:
                            merged_f.write(part_f.read())
                        part_num += 1

                # (2) 병합된 파일 압축 해제
                print("Merging complete. Unzipping...")
                with zipfile.ZipFile(temp_merged_zip, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"Completed Unzip. '{csv_path}' will be used.")

            except Exception as e:
                print(f"Merge/Unzip error: {e}")
                sys.exit(1)
            finally:
                # (3) 임시 병합 파일 삭제
                if os.path.exists(temp_merged_zip):
                    os.remove(temp_merged_zip)

        else:
            print(f"Error: '{csv_filename}' and '{zip_filename}' (or split parts) are not found.")
            print("Please ensure the data file is present.")
            sys.exit(1)

    return csv_path


def plot_apartment_timeseries(unique_id, original_df, reco_df, model, base_date, features, is_best=False):

    if 'font.family' not in plt.rcParams or not plt.rcParams['font.family']:
        set_font()

    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)

    apt_info = reco_df[reco_df['UniqueID'] == unique_id].iloc[0]
    past_df = original_df[original_df['UniqueID'] == unique_id].copy()

    if past_df.empty:
        print(f"경고: UniqueID {unique_id}에 해당하는 과거 거래 데이터가 원본 DF에서 발견되지 않아 시각화를 건너킵니다.")
        return

    # 미래 예측 시계열 데이터 생성
    base_row = past_df.sort_values(by='거래일', ascending=True).iloc[-1].copy()
    start_date = pd.to_datetime(base_row['거래일'])
    end_date = pd.to_datetime('2031-01-01')
    date_range = pd.date_range(start=start_date + pd.offsets.MonthBegin(1), end=end_date, freq='MS')

    future_data = []
    encoded_values = {
        '시도명': base_row['시도명'],
        '시군구명': base_row['시군구명'],
        '법정동': base_row['법정동'],
        '아파트': base_row['아파트'],
        '전용면적': base_row['전용면적'],
        '건축년도': base_row['건축년도'],
	    '기준금리': base_row['기준금리'],
        '가계대출(만원)': base_row['가계대출(만원)'],
        '인구수': base_row['인구수'],
        '실업률': base_row['실업률'],
        'CPI_총지수': base_row['CPI_총지수'],
        'CPI_전년동기': base_row['CPI_전년동기'],
        '월개인소득(만원)': base_row['월개인소득(만원)'],
    }

    for date in date_range:
        encoded_data = encoded_values.copy()
        encoded_data['거래_년'] = date.year
        encoded_data['거래_월'] = date.month
        encoded_data['건축_경과년수'] = date.year - base_row['건축년도']
        encoded_data['최근_거래일_점수'] = (date - base_date).days
        future_data.append(encoded_data)

    if not future_data: return

    future_X = pd.DataFrame(future_data)[features]
    future_prices = model.predict(future_X)

    future_df = pd.DataFrame({
        '거래일': date_range,
        '거래금액': future_prices.astype(int)
    })

    plot_past_df = past_df[['거래일', '거래금액']].copy()
    plot_past_df['거래일'] = pd.to_datetime(plot_past_df['거래일'])

    marker_buy = future_df[future_df['거래일'] == '2026-01-01']
    marker_sell = future_df[future_df['거래일'] == '2030-01-01']

    plt.figure(figsize=(14, 7))
    sns.lineplot(x='거래일', y='거래금액', data=future_df, label='예상 가격 시계열', color='orange', linestyle='--', linewidth=2)
    sns.scatterplot(x='거래일', y='거래금액', data=plot_past_df, label='과거 실제 거래 가격', color='blue', s=50, zorder=5)

    if not marker_buy.empty:
        mbuy = marker_buy.iloc[0]
        plt.scatter(mbuy['거래일'], mbuy['거래금액'], color='red', s=100, zorder=10, label='2026년 1월 매입 예상가')
        plt.annotate(
            f'매입 예상가: {format_manwon(mbuy["거래금액"])}',
            (mbuy['거래일'], mbuy['거래금액']),
            textcoords="offset points",
            xytext=(-30, 15), ha='center', color='red', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )
    if not marker_sell.empty:
        msell = marker_sell.iloc[0]
        plt.scatter(msell['거래일'], msell['거래금액'], color='green', s=100, zorder=10, label='2030년 1월 매각 예상가')
        plt.annotate(
            f'매각 예상가: {format_manwon(msell["거래금액"])}',
            (msell['거래일'], msell['거래금액']),
            textcoords="offset points",
            xytext=(30, 15), ha='center', color='green', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.5)
        )

    title = f"[{apt_info['시도명']} {apt_info['시군구명']} {apt_info['법정동']}] {apt_info['아파트']} ({apt_info['전용면적']:.2f}m²) 가격 시계열 (단위: 만 원)"
    plt.title(title, fontsize=16)
    plt.xlabel("거래일", fontsize=12)
    plt.ylabel("거래 금액 (만 원)", fontsize=12)
    plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format_manwon(x)))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 파일 저장 (이름 규칙 변경)
    # is_best가 True인 경우: TS_BEST_시도명_...png
    # is_best가 False인 경우: TS_시도명_...png (병합 대상)
    prefix = 'TS_BEST' if is_best else 'TS'
    file_name = f"{prefix}_{apt_info['시도명']}_{apt_info['시군구명']}_{apt_info['법정동']}_{apt_info['아파트']}_{apt_info['전용면적']:.2f}m2.png".replace('/', '_').replace('.', '_')
    save_path = os.path.join(PLOT_OUTPUT_DIR, file_name)
    plt.savefig(save_path)
    plt.close()
    return file_name


def combine_images_to_grid(input_dir, output_filename, except_filename, grid_size=(3, 3)):

    print("\n--- Starting to combine plot images into a single grid image ---")


    except_prefix = f"TS_BEST_{except_filename}"
    try:
        image_files = [
            f for f in os.listdir(input_dir)
            if f.startswith('TS_') and not f.startswith(except_prefix) and f.endswith('.png')
        ]
        image_files.sort()
        print(f"Excluding files starting with '{except_prefix}'. {len(image_files)} images found for combination.")
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    if not image_files:
        print("No images found to combine after filtering.")
        return

    try:
        with Image.open(os.path.join(input_dir, image_files[0])) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error opening the first image: {e}")
        return

    cols, rows = grid_size
    num_images_to_combine = min(len(image_files), rows * cols)

    total_width = img_width * cols
    total_height = img_height * rows

    grid_image = Image.new('RGB', (total_width, total_height), 'white')
    print(f"Creating a new {cols}x{rows} grid image with individual size {img_width}x{img_height}...")

    for index, filename in enumerate(image_files):
        if index >= num_images_to_combine:
            break

        row = index // cols
        col = index % cols
        x_offset = col * img_width
        y_offset = row * img_height

        try:
            with Image.open(os.path.join(input_dir, filename)) as img:
                grid_image.paste(img, (x_offset, y_offset))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue


    target_width, target_height = 3000, 1800  # 3x3 그리드에 적합한 가로로 긴 비율
    print(f"Resizing final image to {target_width}x{target_height}...")
    resized_image = grid_image.resize((target_width, target_height), Image.LANCZOS)

    output_path = os.path.join(input_dir, output_filename)
    resized_image.save(output_path, dpi=(150, 150))
    print(f"--- Successfully combined and resized {num_images_to_combine} images into '{output_path}' ---")

    # # 병합에 사용된 개별 파일 삭제
    # print("Deleting temporary individual plot files...")
    # for filename in image_files[:num_images_to_combine]:
    #     os.remove(os.path.join(input_dir, filename))
    # print("Temporary files deleted.")


def predict_region(df, selected_sido, xgb_model, features, cat_features, label_encoders, base_deal_date):

    encoded_sido = label_encoders['시도명'].transform([selected_sido])[0]

    region_df = df[df['시도명'] == encoded_sido].copy()

    all_unique_apts = region_df.drop_duplicates(subset=['UniqueID']).reset_index(drop=True)
    total = len(all_unique_apts)
    print(f"\n[{selected_sido}] 지역의 {total}개 고유 아파트에 대해 예측 데이터 생성 시작.")

    predict_date_buy = pd.to_datetime('2026-01-01')
    predict_date_sell = pd.to_datetime('2030-01-01')

    records = []

    for idx, apt in all_unique_apts.iterrows():
        apt_id = apt['UniqueID']
        past_df = region_df[region_df['UniqueID'] == apt_id].copy()
        if past_df.empty:
            continue

        # 진행상황 표시
        print(f"\r[ {idx + 1} / {total} ] 처리 중...", end='')

        base_row = past_df.sort_values(by='거래일', ascending=True).iloc[-1].copy()

        encoded_values = {
            'UniqueID': apt_id,
            '시도명': base_row['시도명'],
            '시군구명': base_row['시군구명'],
            '법정동': base_row['법정동'],
            '아파트': base_row['아파트'],
            '전용면적': base_row['전용면적'],
            '건축년도': base_row['건축년도'],
            '기준금리': base_row['기준금리'],
            '가계대출(만원)': base_row['가계대출(만원)'],
            '인구수': base_row['인구수'],
            '실업률': base_row['실업률'],
            'CPI_총지수': base_row['CPI_총지수'],
            'CPI_전년동기': base_row['CPI_전년동기'],
            '월개인소득(만원)': base_row['월개인소득(만원)'],
        }

        def make_feature_row(date):
            row = encoded_values.copy()
            row['거래_년'] = date.year
            row['거래_월'] = date.month
            row['건축_경과년수'] = date.year - base_row['건축년도']
            row['최근_거래일_점수'] = (date - base_deal_date).days
            return row

        buy_X = pd.DataFrame([make_feature_row(predict_date_buy)])[features]
        sell_X = pd.DataFrame([make_feature_row(predict_date_sell)])[features]

        buy_price = int(xgb_model.predict(buy_X)[0])
        sell_price = int(xgb_model.predict(sell_X)[0])

        record = encoded_values.copy()
        record['매입예상가_2026_01'] = buy_price
        record['매각예상가_2030_01'] = sell_price
        record['예상_최대이익'] = sell_price - buy_price
        records.append(record)

    print("\n모델 예측 완료. 결과 DataFrame 구성 중...")

    reco_df = pd.DataFrame(records)

    for col in cat_features:
        if col in reco_df.columns:
            reco_df[col] = label_encoders[col].inverse_transform(reco_df[col].astype(int))

    reco_df = reco_df.sort_values(by='예상_최대이익', ascending=False).reset_index(drop=True)
    print(f"[{selected_sido}] 지역 예측 완료. 총 {len(reco_df)}개 결과 생성.")
    return reco_df


if __name__ == "__main__":

    set_font()

    if not os.path.exists(TRAINED_DATA_DIR):
        os.makedirs(TRAINED_DATA_DIR)
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)

    data_dir = './preprocessed'
    data_csv_path = prepare_csv_from_zip(data_dir, 'KoreaApartDeal_PreProcessed.csv', 'KoreaApartDeal_PreProcessed.zip')

    if (os.path.exists(PRELOAD_FILE_PATH) and
            os.path.exists(ENCODERS_FILE_PATH) and
            not SHOULD_RETRAIN):
        print(f"\npreload file found: load '{PRELOAD_FILE_PATH}'.")
        try:
            dtype_spec_loaded = {col: 'int' for col in ['시도명', '시군구명', '법정동', '아파트']}
            df = pd.read_csv(PRELOAD_FILE_PATH, dtype=dtype_spec_loaded)
            df['거래일'] = pd.to_datetime(df['거래일'])
            label_encoders = joblib.load(ENCODERS_FILE_PATH)
            base_deal_date = df['거래일'].min()
            print("Load 데이터 및 LabelEncoder 로드 완료.")
        except Exception as e:
            print(f"load error: {e}. new loading start...")
            SHOULD_RETRAIN = True
    else:
        SHOULD_RETRAIN = True

    if SHOULD_RETRAIN:
        print("\nData load start for XGBoost...")
        try:
            dtype_spec = {'층': 'object'}
            df = pd.read_csv(data_csv_path, dtype=dtype_spec)
        except Exception as e:
            print(f"file read error: {e}")
            sys.exit(1)

        df['거래금액'] = pd.to_numeric(df['거래금액'], errors='coerce')
        df.dropna(subset=['거래금액'], inplace=True)

        core_features = ['거래일', '건축년도', '전용면적', '거래금액', '시도명', '시군구명', '법정동', '아파트', '기준금리', '가계대출(만원)', '인구수', '실업률', 'CPI_총지수', 'CPI_전년동기', '월개인소득(만원)']
        df.dropna(subset=core_features, inplace=True)

        # df['층'] = pd.to_numeric(df['층'], errors='coerce')
        # df.dropna(subset=['층'], inplace=True)
        # df['층'] = df['층'].astype(int)

        df['거래일'] = pd.to_datetime(df['거래일'], errors='coerce')
        df.dropna(subset=['거래일'], inplace=True)
        df['건축년도'] = df['건축년도'].astype(int)

        df['거래_년'] = df['거래일'].dt.year
        df['거래_월'] = df['거래일'].dt.month
        df['건축_경과년수'] = df['거래_년'] - df['건축년도']
        base_deal_date = df['거래일'].min()
        df['최근_거래일_점수'] = (df['거래일'] - base_deal_date).dt.days

        # 최소 거래 횟수 필터링
        apt_counts = df['UniqueID'].value_counts()
        valid_uids = apt_counts[apt_counts >= MIN_TRANSACTION_COUNT].index
        df = df[df['UniqueID'].isin(valid_uids)].copy()
        print(f"최소 {MIN_TRANSACTION_COUNT}회 이상 거래된 아파트로 필터링 후, 남은 거래 기록: {len(df)}개")

        # Label Encoding 수행
        label_encoders = {}
        for col in cat_features:
            df[col] = df[col].astype(str).fillna('missing')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        try:
            df.to_csv(PRELOAD_FILE_PATH, index=False, encoding='utf-8')
            joblib.dump(label_encoders, ENCODERS_FILE_PATH)
            print(f"'{PRELOAD_FILE_PATH}' saved.")
        except Exception as e:
            print(f"Load file error: {e}")

    features = ['시도명', '시군구명', '법정동', '아파트', '전용면적', '건축년도', '거래_년', '거래_월', '건축_경과년수', '최근_거래일_점수', '기준금리', '가계대출(만원)', '인구수', '실업률', 'CPI_총지수', 'CPI_전년동기', '월개인소득(만원)']    
    target = '거래금액'

    # 인코딩된 데이터를 X에 할당
    X = df[features]
    y = df[target]

    if os.path.exists(MODEL_FILE_PATH) and not SHOULD_RETRAIN:
        try:
            xgb_model = joblib.load(MODEL_FILE_PATH)
            print("XGBoost model loaded.")
        except Exception as e:
            print(f"error: {e}. try to train.")
            SHOULD_RETRAIN = True
    else:
        SHOULD_RETRAIN = True

    if SHOULD_RETRAIN:
        print("\nModel Training Started ..")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.3,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        print("XGBoost Model Training Done.")
        try:
            joblib.dump(xgb_model, MODEL_FILE_PATH)
            print(f"Trained Model '{MODEL_FILE_PATH}' is saved")
        except Exception as e:
            print(f"error: {e}")

    decoded_sido = pd.Series(label_encoders['시도명'].inverse_transform(df['시도명'].astype(int)))
    sido_list = sorted(decoded_sido.unique())
    sido_map = {i + 1: sido for i, sido in enumerate(sido_list)}

    while True:
        print("\n" + "=" * 50)
        print("지역 선택: 2026년 1월 매입 기준 2030년 1월 매각시 가장 최대이익을 예측할 지역을 선택해주세요.")
        print("0: 프로그램 종료")
        print("=" * 50)
        for num, sido in sido_map.items():
            print(f"{num}: {sido}")
        print("=" * 50)

        selected_num = None
        try:
            user_input = input("번호를 입력하세요 (0 입력 시 종료): ")
            if user_input.strip() == '':
                continue
            selected_num = int(user_input)
        except ValueError:
            print("유효한 숫자 형식을 입력해주세요.")
            continue

        if selected_num == 0:
            print("\n프로그램을 종료합니다. 감사합니다.")
            sys.exit(0)

        if selected_num not in sido_map:
            print("잘못된 번호입니다. 목록에 있는 번호(1 이상)를 다시 입력해주세요.")
            continue

        selected_sido = sido_map[selected_num]

        reco_df = predict_region(df, selected_sido, xgb_model, features, cat_features, label_encoders, base_deal_date)

        if reco_df.empty:
            print(f"\n경고: {selected_sido} 지역에 대한 예측 결과가 없습니다.")
            continue

        filtered_df = reco_df[reco_df['시도명'] == selected_sido].copy()

        if filtered_df.empty:
            print(f"\n경고: {selected_sido} 지역에 대한 예측 결과가 없습니다. 다른 지역을 선택해주세요.")
            continue

        top_10_apts = filtered_df.head(10).copy()
        best_apt = top_10_apts.iloc[0]


        output_text = "\n" + "=" * 70 + "\n"
        output_text += f"{selected_sido} 최대 이익 아파트 추천 결과 (단위: 만 원)\n"
        output_text += "=" * 70 + "\n"
        output_text += f"**최적 아파트:** {best_apt['아파트']} ({best_apt['시군구명']} {best_apt['법정동']})\n"
        output_text += f"**전용면적:** {best_apt['전용면적']:.2f} m²\n"
        output_text += f"**2025년 12월 예상 매입가:** {format_manwon(best_apt['매입예상가_2026_01'])}\n"
        output_text += f"**2030년 12월 예상 매각가:** {format_manwon(best_apt['매각예상가_2030_01'])}\n"
        output_text += f"**예상 최대 이익 (5년):** {format_manwon(best_apt['예상_최대이익'])}\n"
        output_text += "=" * 70 + "\n"

        output_text += f"\n상위 10개 추천 아파트 목록 ({selected_sido}, 이익 만 원 기준)\n"
        display_cols = ['시도명', '시군구명', '법정동', '아파트', '전용면적', '예상_최대이익', '매입예상가_2026_01', '매각예상가_2030_01']

        top_10_string = top_10_apts[display_cols].to_string(
            index=False,
            formatters={
                '예상_최대이익': '{:,.0f}'.format,
                '매입예상가_2026_01': '{:,.0f}'.format,
                '매각예상가_2030_01': '{:,.0f}'.format,
                '전용면적': '{:.2f}'.format,
            }
        )
        output_text += top_10_string + "\n"
        output_text += "\n" + "=" * 70 + "\n"
        output_text += f"결과는 ./{PLOT_OUTPUT_DIR}/{selected_sido}_APT_Recommendation.txt 파일로 저장되었습니다.\n"
        output_text += "최적 아파트는 개별 PNG 파일로, 나머지 9개는 하나의 PNG 파일로 저장됩니다."
        print(output_text)

        recommendation_file_name = f"{selected_sido}_APT_Recommendation.txt"
        recommendation_file_path = os.path.join(PLOT_OUTPUT_DIR, recommendation_file_name)
        try:
            with open(recommendation_file_path, 'w', encoding='utf-8') as f:
                f.write(output_text)
            print(f"추천 결과가 '{recommendation_file_path}'에 저장되었습니다.")
        except Exception as e:
            print(f"error: {e}")

        # 1. 최대 이익 아파트 (top 1) 개별 저장
        print(f"\n최적 아파트 ({best_apt['아파트']}) 시계열 차트 개별 저장...")
        plot_apartment_timeseries(
            unique_id=best_apt['UniqueID'],
            original_df=df,
            reco_df=reco_df,
            model=xgb_model,
            base_date=base_deal_date,
            features=features,
            is_best=True  # 개별 파일 저장 플래그
        )
        best_apt_filename_prefix = f"{selected_sido}_{best_apt['시군구명']}_{best_apt['법정동']}_{best_apt['아파트']}"

        print(f"\n하위 9개 아파트 시계열 차트 임시 파일 저장 및 병합 시작...")

        temp_files = []

        for i in range(1, len(top_10_apts)):  # 인덱스 1부터 시작 (2번째 아파트부터)
            apt = top_10_apts.iloc[i]
            filename = plot_apartment_timeseries(
                unique_id=apt['UniqueID'],
                original_df=df,
                reco_df=reco_df,
                model=xgb_model,
                base_date=base_deal_date,
                features=features,
                is_best=False  # 병합용 파일 저장 플래그
            )
            if filename:
                temp_files.append(filename)

        if temp_files:
            combine_images_to_grid(
                input_dir=PLOT_OUTPUT_DIR,
                output_filename=f"Combined_Top9_Trends_{selected_sido}.png",
                except_filename=best_apt_filename_prefix,  # TS_BEST_로 시작하는 파일 제외
                grid_size=(3, 3)  # 3x3 그리드
            )
        else:
            print("warn..")

        print("#" * 70)

        input("진행하려면 아무키나 입력하세요")
