# pip install pandas
# pip install numpy
# pip install scikit-learn
# pip install xgboost
# pip install joblib
# joblib 추가 설치 필요

import pandas as pd
import numpy as np
import os
import zipfile
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# --- 설정 변수 ---
MODEL_FILE_PATH = 'xgb_apartment_model.joblib'
# 이 변수를 True로 설정하면 기존 파일이 있어도 무조건 새로 훈련하고 덮어씁니다.
SHOULD_RETRAIN = False


# --------------

# --- 1. 데이터 로딩 및 준비 함수 ---
def prepare_csv_from_zip(data_dir, csv_filename, zip_filename):
    """
    지정된 경로에서 CSV 파일이 없으면 ZIP 파일에서 압축을 해제합니다.
    실패 시 exit(1)로 프로그램 종료.
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
            print(f"Error: '{csv_filename}' and '{zip_filename}' are not found.")
            sys.exit(1)
    return csv_path


# --- 데이터 파일 설정 및 로드 ---
data_dir = './preprocessed'
data_csv_path = prepare_csv_from_zip(data_dir, 'KoreaApartDeal_PreProcessed.csv', 'KoreaApartDeal_PreProcessed.zip')

try:
    dtype_spec = {'층': 'object'}
    df = pd.read_csv(data_csv_path, dtype=dtype_spec)
except Exception as e:
    print(f"file read error: {e}")
    sys.exit(1)

# 금액 오류 해결: '거래금액'을 숫자로 강제 변환
df['거래금액'] = pd.to_numeric(df['거래금액'], errors='coerce')
df.dropna(subset=['거래금액'], inplace=True)

print(f"데이터셋 로드 완료. 총 {len(df)}개 거래 기록. 거래금액(원) 최대값: {df['거래금액'].max():,.0f}")

# --- 2. 데이터 전처리 및 특성 공학 (전체 데이터 사용) ---
print("\n데이터 전처리 및 특성 공학...")

core_features = ['거래일', '건축년도', '전용면적', '층', '거래금액', '시도명', '시군구명', '법정동', '아파트']
df.dropna(subset=core_features, inplace=True)

if df.empty:
    print("경고: 전처리 후 남은 유효한 데이터가 없어 예측을 진행할 수 없습니다.")
    sys.exit(1)

df['층'] = pd.to_numeric(df['층'], errors='coerce')
df.dropna(subset=['층'], inplace=True)
df['층'] = df['층'].astype(int)

df['거래일'] = pd.to_datetime(df['거래일'], errors='coerce')
df.dropna(subset=['거래일'], inplace=True)
df['건축년도'] = df['건축년도'].astype(int)

df['거래_년'] = df['거래일'].dt.year
df['거래_월'] = df['거래일'].dt.month
df['건축_경과년수'] = df['거래_년'] - df['건축년도']

# 컬럼명: 최근_거래일_점수
base_deal_date = df['거래일'].min()
df['최근_거래일_점수'] = (df['거래일'] - base_deal_date).dt.days

cat_features = ['시도명', '시군구명', '법정동', '아파트']
label_encoders = {}
for col in cat_features:
    df[col] = df[col].astype(str).fillna('missing')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features = ['시도명', '시군구명', '법정동', '아파트', '전용면적', '층',
            '건축년도', '거래_년', '거래_월', '건축_경과년수', '최근_거래일_점수']
target = '거래금액'

X = df[features]
y = df[target]

# --- 3. 모델 훈련 및 저장/불러오기 ---
if os.path.exists(MODEL_FILE_PATH) and not SHOULD_RETRAIN:
    # 파일이 있고, 재훈련 플래그가 False일 경우 모델 로드
    print(f"\n모델 파일 발견: '{MODEL_FILE_PATH}'. 모델을 불러옵니다.")
    try:
        xgb_model = joblib.load(MODEL_FILE_PATH)
        print("XGBoost 모델 로드 완료.")
    except Exception as e:
        print(f"모델 파일 로드 오류: {e}. 새로 훈련을 시작합니다.")
        SHOULD_RETRAIN = True
else:
    SHOULD_RETRAIN = True

if SHOULD_RETRAIN:
    # 훈련 시작 (파일이 없거나, 재훈련 플래그가 True일 경우)
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

    # 모델 저장
    try:
        joblib.dump(xgb_model, MODEL_FILE_PATH)
        print(f"훈련된 모델이 '{MODEL_FILE_PATH}'에 저장되었습니다.")
    except Exception as e:
        print(f"모델 저장 오류: {e}")

# --- 4. 전국 대상 예측 시뮬레이션 데이터셋 생성 및 예측 ---
all_unique_apts = df.drop_duplicates(
    subset=['시도명', '시군구명', '법정동', '아파트', '전용면적', '층', '건축년도']
).reset_index(drop=True).copy()

print(f"\n전국 {len(all_unique_apts)}개 고유 아파트에 대해 예측 시뮬레이션 데이터 생성 시작.")

predict_date_2025 = pd.to_datetime('2025-12-01')
predict_date_2030 = pd.to_datetime('2030-12-01')

# 4.1. 예측 데이터셋 벡터화 생성
# buy_X
buy_X_base = all_unique_apts[features].drop(columns=['거래_년', '거래_월', '건축_경과년수', '최근_거래일_점수']).copy()
buy_X = buy_X_base.copy()
buy_X['거래_년'] = predict_date_2025.year
buy_X['거래_월'] = predict_date_2025.month
buy_X['건축_경과년수'] = buy_X['거래_년'] - buy_X['건축년도']
buy_X['최근_거래일_점수'] = (predict_date_2025 - base_deal_date).days

# sell_X (오류 수정: '최근_거래_일_점수' -> '최근_거래일_점수')
sell_X_base = all_unique_apts[features].drop(columns=['거래_년', '거래_월', '건축_경과년수', '최근_거래일_점수']).copy()
sell_X = sell_X_base.copy()
sell_X['거래_년'] = predict_date_2030.year
sell_X['거래_월'] = predict_date_2030.month
sell_X['건축_경과년수'] = sell_X['거래_년'] - sell_X['건축년도']
sell_X['최근_거래일_점수'] = (predict_date_2030 - base_deal_date).days

buy_X = buy_X[features]
sell_X = sell_X[features]

# 4.2. 전국 예측 수행
print("\n모델 예측 수행 시작 (전국)...")
buy_prices_2025 = xgb_model.predict(buy_X)
sell_prices_2030 = xgb_model.predict(sell_X)
print("모델 예측 완료.")

# 4.3. 결과 취합 및 디코딩
reco_df = all_unique_apts[['시도명', '시군구명', '법정동', '아파트', '전용면적', '건축년도']].copy()
reco_df['매입예상가_2025_12'] = buy_prices_2025.astype(int)
reco_df['매각예상가_2030_12'] = sell_prices_2030.astype(int)
reco_df['예상_최대이익'] = reco_df['매각예상가_2030_12'] - reco_df['매입예상가_2025_12']

for col in cat_features:
    reco_df[col] = label_encoders[col].inverse_transform(reco_df[col])

reco_df = reco_df.sort_values(by='예상_최대이익', ascending=False).reset_index(drop=True)


# 금액 포맷팅 함수 정의 (세 자리 쉼표 추가)
def format_won(amount):
    return f"{amount:,.0f} 원"


# 5.1. 시도명 목록 생성
sido_list = sorted(reco_df['시도명'].unique())
sido_map = {i + 1: sido for i, sido in enumerate(sido_list)}

# --- 5. 사용자 입력 기반 동적 필터링 및 출력 (무한 반복) ---
while True:
    print("\n" + "=" * 50)
    print("지역 선택: 예측 결과를 볼 시도명을 선택해주세요.")
    print("0: 프로그램 종료")
    print("=" * 50)
    for num, sido in sido_map.items():
        print(f"{num}: {sido}")
    print("=" * 50)

    # 5.2. 사용자 입력 받기
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

    # 5.3. 결과 출력
    best_apt = filtered_df.iloc[0]

    print("\n" + "=" * 70)
    print(f"{selected_sido} 최대 이익 아파트 추천 결과 (원 단위, 세 자리 쉼표)")
    print("=" * 70)
    print(f"**최적 아파트:** {best_apt['아파트']} ({best_apt['시군구명']} {best_apt['법정동']})")
    print(f"**전용면적:** {best_apt['전용면적']:.2f} m² | **건축년도:** {best_apt['건축년도']}")
    print(f"**2025년 12월 예상 매입가:** {format_won(best_apt['매입예상가_2025_12'])}")
    print(f"**2030년 12월 예상 매각가:** {format_won(best_apt['매각예상가_2030_12'])}")
    print(f"**예상 최대 이익 (5년):** {format_won(best_apt['예상_최대이익'])}")
    print("=" * 70)

    # 상위 10개 추천 목록 출력 (원 단위)
    print(f"\n상위 10개 추천 아파트 목록 ({selected_sido}, 이익 원 기준)")
    display_cols = ['시도명', '시군구명', '법정동', '아파트', '전용면적', '예상_최대이익', '매입예상가_2025_12', '매각예상가_2030_12']

    print(filtered_df[display_cols].head(10).to_string(
        index=False,
        formatters={
            '예상_최대이익': '{:,.0f}'.format,
            '매입예상가_2025_12': '{:,.0f}'.format,
            '매각예상가_2030_12': '{:,.0f}'.format,
            '전용면적': '{:.2f}'.format
        }
    ))