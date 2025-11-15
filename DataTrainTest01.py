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

# --- 설정 변수 ---
# 모델 및 전처리 파일 저장 폴더
TRAINED_DATA_DIR = 'trained_data'
MODEL_FILE_PATH = os.path.join(TRAINED_DATA_DIR, 'xgb_apartment_model.joblib')
PRELOAD_FILE_PATH = os.path.join(TRAINED_DATA_DIR, 'preload_xgb_data.csv')
ENCODERS_FILE_PATH = os.path.join(TRAINED_DATA_DIR, 'label_encoders.joblib')
PLOT_OUTPUT_DIR = 'results'  # 그래프 및 추천 결과 저장 폴더

# 제어 플래그
SHOULD_RETRAIN = True  # 모델을 다시 훈련할지 여부 (최초 실행 시 True 권장)
MIN_TRANSACTION_COUNT = 10  # 최소 거래 횟수 필터링 기준


# --------------

# **********************************************
# ** Matplotlib OS별 한글 폰트 설정 함수 **
# **********************************************
def set_font():
    """OS에 따라 적절한 한글 폰트를 설정하고, 음수 부호 깨짐을 방지합니다."""
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

# --- 금액 포맷팅 함수 정의 (단위: 만 원) ---
def format_manwon(amount):
    """금액을 세 자리 쉼표와 '만 원'을 포함하여 포맷팅합니다."""
    # amount가 넘파이 float 타입일 수 있으므로 int로 변환
    if pd.isna(amount):
        return "N/A"
    return f"{int(amount):,.0f} 만 원"


# --- 1. 데이터 로딩 및 준비 함수 ---
def prepare_csv_from_zip(data_dir, csv_filename, zip_filename):
    """
    지정된 경로에서 CSV 파일이 없으면 ZIP 파일에서 압축을 해제합니다.
    """
    csv_path = os.path.join(data_dir, csv_filename)
    zip_path = os.path.join(data_dir, zip_filename)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(csv_path):
        print(f"'{csv_path}' is not exist. Checking Zip file.")
        if os.path.exists(zip_path):
            print(f"'{zip_path}' is found. Unzip...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"Completed Unzip. '{csv_path}' will be used.")
            except Exception as e:
                print(f"Unzip error: {e}")
                sys.exit(1)
        else:
            print(f"Error: '{csv_filename}' and '{zip_filename}' are not found. Please ensure the data file is present.")
            sys.exit(1)
    return csv_path


# --- 2. 시계열 데이터 준비 및 시각화 함수 ---
def plot_apartment_timeseries(unique_id, original_df, reco_df, model, base_date, features, is_best=False):
    """
    특정 UniqueID를 가진 아파트의 과거 거래 데이터와 미래 예측 가격을 시계열로 시각화하고 파일로 저장합니다. (단위: 만 원)
    is_best=True면 개별 파일, False면 병합용 파일 이름 규칙을 따릅니다.
    """
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

    marker_2025 = future_df[future_df['거래일'] == '2025-12-01']
    marker_2030 = future_df[future_df['거래일'] == '2030-12-01']

    # 플롯 설정
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='거래일', y='거래금액', data=future_df, label='예상 가격 시계열', color='orange', linestyle='--', linewidth=2)
    sns.scatterplot(x='거래일', y='거래금액', data=plot_past_df, label='과거 실제 거래 가격', color='blue', s=50, zorder=5)

    # 주요 예측 지점 표시
    if not marker_2025.empty:
        m25 = marker_2025.iloc[0]
        plt.scatter(m25['거래일'], m25['거래금액'], color='red', s=100, zorder=10, label='2025년 12월 매입 예상가')
        plt.annotate(
            f'매입 예상가: {format_manwon(m25["거래금액"])}',
            (m25['거래일'], m25['거래금액']),
            textcoords="offset points",
            xytext=(-30, 15), ha='center', color='red', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
        )
    if not marker_2030.empty:
        m30 = marker_2030.iloc[0]
        plt.scatter(m30['거래일'], m30['거래금액'], color='green', s=100, zorder=10, label='2030년 12월 매각 예상가')
        plt.annotate(
            f'매각 예상가: {format_manwon(m30["거래금액"])}',
            (m30['거래일'], m30['거래금액']),
            textcoords="offset points",
            xytext=(30, 15), ha='center', color='green', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.5)
        )

    # 시각화 설정
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
    plt.close()  # 메모리 해제
    return file_name  # 병합을 위해 파일 이름 반환


# **********************************************
# ** 이미지 병합 함수 (요청하신 함수 구조를 따름) **
# **********************************************
def combine_images_to_grid(input_dir, output_filename, except_filename, grid_size=(3, 3)):
    """
    특정 패턴의 이미지 파일들을 그리드 형태로 합쳐 하나의 PNG 파일로 저장합니다.
    (하위 9개 아파트를 3x3 그리드로 합치기 위해 grid_size=(3, 3)으로 수정)
    """
    print("\n--- Starting to combine plot images into a single grid image ---")

    # 1. 'TS'로 시작하고 'except_filename'을 포함하지 않는 파일을 가져옵니다.
    except_prefix = f"TS_BEST_{except_filename}"  # 베스트 아파트 파일 이름 접두사
    try:
        # PLOT_OUTPUT_DIR에서 'TS_'로 시작하고 'TS_BEST_'로 시작하지 않는 파일만 가져옵니다.
        image_files = [
            f for f in os.listdir(input_dir)
            if f.startswith('TS_') and not f.startswith(except_prefix) and f.endswith('.png')
        ]
        image_files.sort()  # 파일 이름을 기준으로 정렬
        print(f"Excluding files starting with '{except_prefix}'. {len(image_files)} images found for combination.")
    except FileNotFoundError:
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    if not image_files:
        print("No images found to combine after filtering.")
        return

    # 첫 번째 이미지를 열어 개별 이미지의 크기를 확인합니다.
    try:
        with Image.open(os.path.join(input_dir, image_files[0])) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Error opening the first image: {e}")
        return

    cols, rows = grid_size
    num_images_to_combine = min(len(image_files), rows * cols)

    # 9개 (3x3)만 합치므로, 전체 그리드 크기는 3x3에 맞춥니다.
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

    # 9개의 이미지를 합쳤으므로, 최종 이미지를 보기 좋게 리사이즈합니다.
    # 3x3 비율(42:21)에 맞는 3000x2100 (14:10) 또는 3000x1800 (15:9) 비율로 조정
    target_width, target_height = 3000, 1800  # 3x3 그리드에 적합한 가로로 긴 비율
    print(f"Resizing final image to {target_width}x{target_height}...")
    # Image.Resampling.LANCZOS 대신 Image.LANCZOS 사용 (PIL 버전 호환성)
    resized_image = grid_image.resize((target_width, target_height), Image.LANCZOS)

    output_path = os.path.join(input_dir, output_filename)
    resized_image.save(output_path, dpi=(150, 150))
    print(f"--- Successfully combined and resized {num_images_to_combine} images into '{output_path}' ---")

    # 병합에 사용된 개별 파일 삭제
    print("Deleting temporary individual plot files...")
    for filename in image_files[:num_images_to_combine]:
        os.remove(os.path.join(input_dir, filename))
    print("Temporary files deleted.")


# **********************************************
# ** 메인 프로그램 시작 **
# **********************************************

# 폰트 설정 실행
set_font()

# trained_data 및 results 폴더 생성
if not os.path.exists(TRAINED_DATA_DIR):
    os.makedirs(TRAINED_DATA_DIR)
if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)

# --- 데이터 파일 설정 및 로드 ---
data_dir = './preprocessed'
data_csv_path = prepare_csv_from_zip(data_dir, 'KoreaApartDeal_PreProcessed.csv', 'KoreaApartDeal_PreProcessed.zip')

# --- 3. 데이터 전처리 및 특성 공학 (저장/로드 로직) ---
# UniqueID를 생성하는 코드가 원본 코드에 포함되어 있지 않아, 원본 데이터 로드 시점(SHOULD_RETRAIN == True)에 수행하는 것이 안전합니다.

if (os.path.exists(PRELOAD_FILE_PATH) and
        os.path.exists(ENCODERS_FILE_PATH) and
        not SHOULD_RETRAIN):
    # 로드 로직
    # ... (생략: 이전 코드와 동일)
    print(f"\n전처리된 파일 발견: '{PRELOAD_FILE_PATH}'. 파일을 로드합니다.")
    try:
        dtype_spec_loaded = {col: 'int' for col in ['시도명', '시군구명', '법정동', '아파트']}
        df = pd.read_csv(PRELOAD_FILE_PATH, dtype=dtype_spec_loaded)
        df['거래일'] = pd.to_datetime(df['거래일'])
        label_encoders = joblib.load(ENCODERS_FILE_PATH)
        base_deal_date = df['거래일'].min()
        print("전처리된 데이터 및 LabelEncoder 로드 완료.")
    except Exception as e:
        print(f"파일 로드 오류: {e}. 새로 전처리를 시작합니다.")
        SHOULD_RETRAIN = True
else:
    SHOULD_RETRAIN = True

if SHOULD_RETRAIN:
    print("\nXGBoost를 위한 데이터 전처리 시작...")
    try:
        dtype_spec = {'층': 'object'}
        df = pd.read_csv(data_csv_path, dtype=dtype_spec)
    except Exception as e:
        print(f"file read error: {e}")
        sys.exit(1)

    df['거래금액'] = pd.to_numeric(df['거래금액'], errors='coerce')
    df.dropna(subset=['거래금액'], inplace=True)

    core_features = ['거래일', '건축년도', '전용면적', '거래금액', '시도명', '시군구명', '법정동', '아파트']
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

    # **UniqueID 생성 (필수)**
    # df['UniqueID'] = df.apply(
    #     lambda row: f"{row['시도명']}_{row['시군구명']}_{row['법정동']}_{row['아파트']}_{row['전용면적']:.2f}_{row['건축년도']}}",
    #     axis=1
    # )

    # 최소 거래 횟수 필터링
    apt_counts = df['UniqueID'].value_counts()
    valid_uids = apt_counts[apt_counts >= MIN_TRANSACTION_COUNT].index
    df = df[df['UniqueID'].isin(valid_uids)].copy()
    print(f"최소 {MIN_TRANSACTION_COUNT}회 이상 거래된 아파트로 필터링 후, 남은 거래 기록: {len(df)}개")

    # Label Encoding 수행
    cat_features = ['시도명', '시군구명', '법정동', '아파트']
    label_encoders = {}
    for col in cat_features:
        df[col] = df[col].astype(str).fillna('missing')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 전처리 완료된 파일 저장
    try:
        df.to_csv(PRELOAD_FILE_PATH, index=False, encoding='utf-8')
        joblib.dump(label_encoders, ENCODERS_FILE_PATH)
        print(f"전처리된 데이터와 인코더가 '{TRAINED_DATA_DIR}'에 저장되었습니다.")
    except Exception as e:
        print(f"전처리 파일 저장 오류: {e}")

features = ['시도명', '시군구명', '법정동', '아파트', '전용면적',
            '건축년도', '거래_년', '거래_월', '건축_경과년수', '최근_거래일_점수']  # 층을 빼고 예측에 사용 (UniqueID에 포함됨)
target = '거래금액'

# 인코딩된 데이터를 X에 할당
X = df[features]
y = df[target]

# --- 4. 모델 훈련 및 저장/불러오기 ---
if os.path.exists(MODEL_FILE_PATH) and not SHOULD_RETRAIN:
    try:
        xgb_model = joblib.load(MODEL_FILE_PATH)
        print("XGBoost 모델 로드 완료.")
    except Exception as e:
        print(f"모델 파일 로드 오류: {e}. 새로 훈련을 시작합니다.")
        SHOULD_RETRAIN = True
else:
    SHOULD_RETRAIN = True

if SHOULD_RETRAIN:
    print("\n모델 훈련 시작 (새로운 훈련/재훈련)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost 모델 훈련 완료.")
    try:
        joblib.dump(xgb_model, MODEL_FILE_PATH)
        print(f"훈련된 모델이 '{MODEL_FILE_PATH}'에 저장되었습니다.")
    except Exception as e:
        print(f"모델 저장 오류: {e}")

# --- 5. 전국 대상 예측 시뮬레이션 데이터셋 생성 및 예측 ---
all_unique_apts = df.drop_duplicates(
    subset=['UniqueID']
).reset_index(drop=True).copy()
print(f"\n전국 {len(all_unique_apts)}개 고유 아파트에 대해 예측 시뮬레이션 데이터 생성 시작.")

predict_date_2025 = pd.to_datetime('2025-12-01')
predict_date_2030 = pd.to_datetime('2030-12-01')

base_cols = ['UniqueID', '시도명', '시군구명', '법정동', '아파트', '전용면적', '건축년도']
buy_X = all_unique_apts[base_cols].copy()
buy_X['거래_년'] = predict_date_2025.year
buy_X['거래_월'] = predict_date_2025.month
buy_X['건축_경과년수'] = buy_X['거래_년'] - buy_X['건축년도']
buy_X['최근_거래일_점수'] = (predict_date_2025 - base_deal_date).days

sell_X = all_unique_apts[base_cols].copy()
sell_X['거래_년'] = predict_date_2030.year
sell_X['거래_월'] = predict_date_2030.month
sell_X['건축_경과년수'] = sell_X['거래_년'] - sell_X['건축년도']
sell_X['최근_거래일_점수'] = (predict_date_2030 - base_deal_date).days

buy_X_model = buy_X[features]
sell_X_model = sell_X[features]

print("\n모델 예측 수행 시작 (전국)...")
buy_prices_2025 = xgb_model.predict(buy_X_model)
sell_prices_2030 = xgb_model.predict(sell_X_model)
print("모델 예측 완료.")

reco_df = all_unique_apts[['UniqueID', '시도명', '시군구명', '법정동', '아파트', '전용면적', '건축년도']].copy()
reco_df['매입예상가_2025_12'] = buy_prices_2025.astype(int)
reco_df['매각예상가_2030_12'] = sell_prices_2030.astype(int)
reco_df['예상_최대이익'] = reco_df['매각예상가_2030_12'] - reco_df['매입예상가_2025_12']

for col in cat_features:
    reco_df[col] = label_encoders[col].inverse_transform(reco_df[col].astype(int))

reco_df = reco_df.sort_values(by='예상_최대이익', ascending=False).reset_index(drop=True)

sido_list = sorted(reco_df['시도명'].unique())
sido_map = {i + 1: sido for i, sido in enumerate(sido_list)}

# --- 7. 사용자 입력 기반 동적 필터링 및 출력 (무한 반복) ---
while True:
    print("\n" + "=" * 50)
    print("지역 선택: 예측 결과를 볼 **시도명**을 선택해주세요.")
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
    filtered_df = reco_df[reco_df['시도명'] == selected_sido].copy()

    if filtered_df.empty:
        print(f"\n경고: {selected_sido} 지역에 대한 예측 결과가 없습니다. 다른 지역을 선택해주세요.")
        continue

    # 7.3. 결과 출력 및 파일 저장 준비
    top_10_apts = filtered_df.head(10).copy()
    best_apt = top_10_apts.iloc[0]

    # 출력 내용 문자열로 구성
    output_text = "\n" + "=" * 70 + "\n"
    output_text += f"🏠 {selected_sido} 최대 이익 아파트 추천 결과 (단위: 만 원)\n"
    output_text += "=" * 70 + "\n"
    output_text += f"**최적 아파트:** {best_apt['아파트']} ({best_apt['시군구명']} {best_apt['법정동']})\n"
    output_text += f"**전용면적:** {best_apt['전용면적']:.2f} m²\n"
    output_text += f"**2025년 12월 예상 매입가:** {format_manwon(best_apt['매입예상가_2025_12'])}\n"
    output_text += f"**2030년 12월 예상 매각가:** {format_manwon(best_apt['매각예상가_2030_12'])}\n"
    output_text += f"**예상 최대 이익 (5년):** {format_manwon(best_apt['예상_최대이익'])}\n"
    output_text += "=" * 70 + "\n"

    output_text += f"\n상위 10개 추천 아파트 목록 ({selected_sido}, 이익 만 원 기준)\n"
    display_cols = ['시도명', '시군구명', '법정동', '아파트', '전용면적', '예상_최대이익', '매입예상가_2025_12', '매각예상가_2030_12']

    top_10_string = top_10_apts[display_cols].to_string(
        index=False,
        formatters={
            '예상_최대이익': '{:,.0f}'.format,
            '매입예상가_2025_12': '{:,.0f}'.format,
            '매각예상가_2030_12': '{:,.0f}'.format,
            '전용면적': '{:.2f}'.format,
        }
    )
    output_text += top_10_string + "\n"
    output_text += "\n" + "=" * 70 + "\n"
    output_text += f"결과는 ./{PLOT_OUTPUT_DIR}/{selected_sido}_Apt_Recommendation.txt 파일로 저장되었습니다.\n"
    output_text += "최적 아파트는 개별 PNG 파일로, 나머지 9개는 하나의 PNG 파일로 저장됩니다."
    # 콘솔 출력
    print(output_text)

    # 파일 저장
    recommendation_file_name = f"{selected_sido}_Apt_Recommendation.txt"
    recommendation_file_path = os.path.join(PLOT_OUTPUT_DIR, recommendation_file_name)
    try:
        with open(recommendation_file_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"추천 결과가 '{recommendation_file_path}'에 저장되었습니다.")
    except Exception as e:
        print(f"경고: 추천 결과 파일 저장 오류: {e}")

    # 7.4. 상위 10개 아파트 시계열 시각화 함수 호출 및 병합 로직

    # 1. 최대 이익 아파트 (top 1) 개별 저장
    print(f"\n[1/2] 최적 아파트 ({best_apt['아파트']}) 시계열 차트 개별 저장...")
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

    # 2. 하위 9개 아파트 (top 2 ~ 10) 임시 파일로 저장
    print(f"\n[2/2] 하위 9개 아파트 시계열 차트 임시 파일 저장 및 병합 시작...")

    # 임시 파일 목록을 추적합니다. (combine_images_to_grid 함수에서 삭제될 예정)
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

    # 3. 임시 파일들을 그리드로 병합
    if temp_files:
        combine_images_to_grid(
            input_dir=PLOT_OUTPUT_DIR,
            output_filename=f"Combined_Top9_Trends_{selected_sido}.png",
            except_filename=best_apt_filename_prefix,  # TS_BEST_로 시작하는 파일 제외
            grid_size=(3, 3)  # 3x3 그리드
        )
    else:
        print("경고: 하위 9개 아파트 중 시계열 차트를 생성할 유효한 데이터가 충분하지 않아 이미지 병합을 건너뜁니다.")

    print("#" * 70)