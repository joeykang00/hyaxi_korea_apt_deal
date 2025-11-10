import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

# --- 1. 파일 준비 (이전과 동일) ---
def prepare_csv_from_zip(data_dir, csv_filename, zip_filename):
    csv_path = os.path.join(data_dir, csv_filename)
    zip_path = os.path.join(data_dir, zip_filename)
    if not os.path.exists(csv_path):
        print(f"'{csv_path}' 파일이 없습니다. 압축 파일을 확인합니다.")
        if os.path.exists(zip_path):
            print(f"'{zip_path}' 파일을 발견했습니다. 압축을 해제합니다...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"압축 해제가 완료되었습니다. '{csv_path}' 파일을 사용합니다.")
            except Exception as e:
                print(f"압축 해제 중 오류가 발생했습니다: {e}")
                exit()
        else:
            print(f"오류: '{csv_filename}'과 '{zip_filename}'을 모두 찾을 수 없습니다.")
            exit()
    return csv_path

data_dir = './data'
deal_csv_path = prepare_csv_from_zip(data_dir, 'KoreaApartDeal.csv', 'KoreaApartDeal.zip')
loc_csv_path = prepare_csv_from_zip(data_dir, 'LocationCode.csv', 'LocationCode.zip')


# --- 2. 데이터 불러오기 (이전과 동일) ---
try:
    deal_df = pd.read_csv(deal_csv_path)
    loc_df = pd.read_csv(loc_csv_path, dtype={'법정동코드': str, '읍면동명': str, '리명': str})
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    exit()


# --- 2.5. [신규] 필수 데이터 결측치 제거 ---
print(f"\n초기 데이터 개수: {len(deal_df):,}개")
# '거래일' 또는 '거래금액'에 값이 없는 행을 제거
initial_rows = len(deal_df)
deal_df.dropna(subset=['거래일', '거래금액'], inplace=True)
final_rows = len(deal_df)
print(f"'거래일' 또는 '거래금액'이 없는 데이터 {initial_rows - final_rows:,}개를 제거했습니다.")
print(f"처리 후 데이터 개수: {final_rows:,}개")


# --- 3. ~ 7. (기존 코드와 동일, 번호만 조정) ---
loc_df_filtered = loc_df[loc_df['시군구명'].notna()].copy()
loc_df_filtered['지역코드'] = loc_df_filtered['법정동코드'].str[:5].astype(int)
loc_map = loc_df_filtered[['지역코드', '시도명', '시군구명']].drop_duplicates()
df = pd.merge(deal_df, loc_map, on='지역코드', how='left')

loc_lookup = loc_df[['시도명', '시군구명', '읍면동명', '리명', '법정동코드']].copy()
loc_lookup['법정동'] = (loc_lookup['읍면동명'].fillna('') + ' ' + loc_lookup['리명'].fillna('')).str.strip()
final_df = pd.merge(df, loc_lookup[['시도명', '시군구명', '법정동', '법정동코드']],
                    on=['시도명', '시군구명', '법정동'],
                    how='left')

unique_apart_count = final_df['아파트'].nunique()
print(f"\n고유한 아파트 이름의 개수: {unique_apart_count:,}개")
final_df['아파트ID'] = pd.factorize(final_df['아파트'])[0]
print("'아파트ID' 컬럼이 성공적으로 추가되었습니다.")

final_df['아파트ID'] = final_df['아파트ID'].astype(str).str.zfill(5)
print("'아파트ID'를 5자리 문자열로 포맷팅했습니다.")
final_df['전용면적ID'] = final_df['전용면적'].round(0).astype(int).astype(str).str.zfill(3)
print("'전용면적ID'를 3자리 문자열로 포맷팅하고 새로 추가했습니다.")

final_df['UniqueID'] = final_df['법정동코드'] + final_df['아파트ID'] + final_df['전용면적ID']
print("\n'UniqueID' 컬럼이 성공적으로 추가되었습니다.")


# --- 8. 결과 확인 및 컬럼 순서 정리 (이전과 동일) ---
final_columns = [
    'UniqueID', '시도명', '시군구명', '법정동', '아파트',
    '전용면적', '거래일', '거래금액',
]
final_columns_exist = [col for col in final_columns if col in final_df.columns]
final_df = final_df[final_columns_exist]

print("\n'UniqueID' 추가 후 최종 결과 (상위 5개):")
print(final_df.head())


# --- 9. 최종 결과 파일 저장 (이전과 동일) ---
output_path = os.path.join(data_dir, 'KoreaApartDeal_Final.csv')
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n최종 결과 파일이 {output_path} 로 저장되었습니다.")

# --- 10. [수정] 시각화 ---

print("\n--- 데이터 시각화 시작 ---")

# 1. 시각화를 위한 데이터 준비 (이전과 동일)
final_df['거래금액'] = final_df['거래금액'].astype(str).str.strip()
final_df['거래금액'] = pd.to_numeric(final_df['거래금액'], errors='coerce')
final_df.dropna(subset=['거래금액'], inplace=True)
final_df['거래금액'] = final_df['거래금액'].astype(int)
final_df['거래일'] = pd.to_datetime(final_df['거래일'], errors='coerce')
final_df.dropna(subset=['거래일'], inplace=True)

# 2. 한글 폰트 설정 (이전과 동일)
import matplotlib.font_manager as fm

try:
    font_path = fm.findfont('NanumGothic', fallback_to_default=False)
    plt.rc('font', family='NanumGothic')
    print("'NanumGothic' 폰트가 성공적으로 설정되었습니다.")
except:
    print("경고: 'NanumGothic' 폰트를 찾을 수 없습니다. 한글이 깨질 수 있습니다.")
plt.rcParams['axes.unicode_minus'] = False


# 3. [수정] 시도명별로 그래프를 생성하고 PNG 파일로 저장하는 함수
def save_plots_by_sido(df, output_dir='apartment_price_graphs'):
    """
    데이터프레임을 '시도명'으로 그룹화하여 각 지역별로 모든 UniqueID의
    가격 추이 그래프를 생성하고, 지정된 디렉터리에 PNG 파일로 저장합니다.
    """
    # 1. 결과를 저장할 디렉터리 생성 (없으면 새로 만듦)
    os.makedirs(output_dir, exist_ok=True)
    print(f"그래프가 '{output_dir}' 폴더에 저장됩니다.")

    # 2. 전체 시도명 목록 가져오기
    sido_list = df['시도명'].unique()

    # 3. 각 시도명에 대해 반복 작업
    for sido in sido_list:
        print(f"\n'{sido}' 지역의 그래프를 생성 중입니다...")

        # 현재 시도에 해당하는 데이터만 필터링
        sido_df = df[df['시도명'] == sido].copy()

        if sido_df.empty:
            print(f"'{sido}' 지역에 데이터가 없어 건너뜁니다.")
            continue

        # 4. 그래프 생성 시작
        plt.figure(figsize=(15, 8))
        ax = plt.gca()

        # 5. 해당 지역의 모든 UniqueID에 대해 선 그래프 그리기
        unique_ids_in_sido = sido_df['UniqueID'].unique()
        for unique_id in unique_ids_in_sido:
            target_df = sido_df[sido_df['UniqueID'] == unique_id].copy()
            if not target_df.empty:
                target_df.sort_values('거래일', inplace=True)
                # 점(marker) 없이 선(line)만 그리기
                ax.plot(target_df['거래일'], target_df['거래금액'], marker='', linestyle='-', alpha=0.5)

        # 6. 그래프 서식 설정
        plt.title(f"'{sido}' 전체 아파트 가격 추이", fontsize=16)
        plt.xlabel("거래일", fontsize=12)
        plt.ylabel("거래금액 (만원)", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # 7. 그래프를 PNG 파일로 저장
        # 파일명에 특수문자가 포함될 경우를 대비하여 안전하게 처리
        safe_sido_name = "".join(c for c in sido if c.isalnum())
        filename = f"{safe_sido_name}_가격추이.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=150)  # dpi 옵션으로 해상도 조절 가능
        print(f"'{filepath}' 파일로 저장되었습니다.")

        # 8. 메모리 관리를 위해 현재 그래프 닫기 (매우 중요)
        plt.close()

    print("\n--- 모든 지역의 그래프 생성이 완료되었습니다. ---")


# 4. [수정] 예시 실행
if not final_df.dropna(subset=['UniqueID', '시도명']).empty:
    # 'apartment_price_graphs' 라는 폴더에 시도별 그래프가 저장됩니다.
    save_plots_by_sido(final_df, output_dir='apartment_price_graphs')
else:
    print("\n분석할 데이터가 없어 그래프를 생성할 수 없습니다.")