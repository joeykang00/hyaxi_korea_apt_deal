# pip install pandas matplotlib

import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

def prepare_csv_from_zip(data_dir, csv_filename, zip_filename):
    csv_path = os.path.join(data_dir, csv_filename)
    zip_path = os.path.join(data_dir, zip_filename)
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
                exit()
        else:
            print(f"Error: '{csv_filename}' and '{zip_filename}'are not found.")
            exit()
    return csv_path

data_dir = './data'
deal_csv_path = prepare_csv_from_zip(data_dir, 'KoreaApartDeal.csv', 'KoreaApartDeal.zip')
loc_csv_path = prepare_csv_from_zip(data_dir, 'LocationCode.csv', 'LocationCode.zip')

try:
    deal_df = pd.read_csv(deal_csv_path)
    location_df = pd.read_csv(loc_csv_path, dtype={'법정동코드': str, '읍면동명': str, '리명': str})
except Exception as e:
    print(f"file read error: {e}")
    exit()


print(f"\n# of Total Data: {len(deal_df):,}")
initial_rows = len(deal_df)
deal_df.dropna(subset=['거래일', '거래금액'], inplace=True)
final_rows = len(deal_df)
print(f"'거래일' 또는 '거래금액'이 없는 데이터 {initial_rows - final_rows:,}개를 제거했습니다.")
print(f"Number of Processing : {final_rows:,}개")


location_df_filtered = location_df[location_df['시군구명'].notna()].copy()
location_df_filtered['지역코드'] = location_df_filtered['법정동코드'].str[:5].astype(int)
loc_map = location_df_filtered[['지역코드', '시도명', '시군구명']].drop_duplicates()
df = pd.merge(deal_df, loc_map, on='지역코드', how='left')

loc_lookup = location_df[['시도명', '시군구명', '읍면동명', '리명', '법정동코드']].copy()
loc_lookup['법정동'] = (loc_lookup['읍면동명'].fillna('') + ' ' + loc_lookup['리명'].fillna('')).str.strip()
final_df = pd.merge(df, loc_lookup[['시도명', '시군구명', '법정동', '법정동코드']],
                    on=['시도명', '시군구명', '법정동'],
                    how='left')

unique_apart_count = final_df['아파트'].nunique()
final_df['아파트ID'] = pd.factorize(final_df['아파트'])[0]
final_df['아파트ID'] = final_df['아파트ID'].astype(str).str.zfill(5)
final_df['전용면적ID'] = final_df['전용면적'].round(0).astype(int).astype(str).str.zfill(3)
final_df['UniqueID'] = final_df['법정동코드'] + final_df['아파트ID'] + final_df['전용면적ID']

final_columns = [
    'UniqueID', '시도명', '시군구명', '법정동', '아파트',
    '전용면적', '거래일', '거래금액',
]
final_columns_exist = [col for col in final_columns if col in final_df.columns]
final_df = final_df[final_columns_exist]
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
print("\n'UniqueID' and final (head 5th):")
print(final_df.head())

output_dir='preprocessed'
output_file='KoreaApartDeal_PreProcessed.csv'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\nPre-Processed File {output_path} is saved.")


print("\n--- Starting Data Visualization ---")

final_df['거래금액'] = final_df['거래금액'].astype(str).str.strip()
final_df['거래금액'] = pd.to_numeric(final_df['거래금액'], errors='coerce')
final_df.dropna(subset=['거래금액'], inplace=True)
final_df['거래금액'] = final_df['거래금액'].astype(int)
final_df['거래일'] = pd.to_datetime(final_df['거래일'], errors='coerce')
final_df.dropna(subset=['거래일'], inplace=True)

try:
    font_path = fm.findfont('NanumGothic', fallback_to_default=False)
    plt.rc('font', family='NanumGothic')
    print("'NanumGothic' is set successfully.")
except:
    print("경고: 'NanumGothic' is not found. Hangul can be displayed corruptly.")
plt.rcParams['axes.unicode_minus'] = False


def save_top100_plots_by_sido(df, output_dir='preprocessed'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"transcation top 100 chart images are saved in '{output_dir}' folder.")

    # 2. 전체 시도명 목록 가져오기
    sido_list = df['시도명'].unique()

    # 3. 각 시도명에 대해 반복 작업
    for sido in sido_list:
        print(f"\n'{sido}' chart is saving...")

        # 현재 시도에 해당하는 데이터만 필터링
        sido_df = df[df['시도명'] == sido].copy()

        if sido_df.empty:
            print(f"'{sido}' is empty. Skip!")
            continue

        transaction_counts = sido_df['UniqueID'].value_counts()
        top_100_ids = transaction_counts.head(100).index
        top_100_df = sido_df[sido_df['UniqueID'].isin(top_100_ids)]

        print(f"Graphing the top 100 apartment transactions out of a total of {len(transaction_counts):,} in the '{sido}' region.")

        plt.figure(figsize=(15, 8))
        ax = plt.gca()

        for unique_id in top_100_ids:
            target_df = top_100_df[top_100_df['UniqueID'] == unique_id].copy()
            if not target_df.empty:
                target_df.sort_values('거래일', inplace=True)
                ax.plot(target_df['거래일'], target_df['거래금액'], marker='', linestyle='-', alpha=0.5)

        plt.title(f"'{sido}' 거래량 상위 100개 아파트 가격 추이", fontsize=16)
        plt.xlabel("거래일", fontsize=12)
        plt.ylabel("거래금액 (만원)", fontsize=12)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        safe_sido_name = "".join(c for c in sido if c.isalnum())
        filename = f"{safe_sido_name}_가격추이_상위100.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=300)
        print(f"'{filepath}' file saved.")

        plt.close()

    print("\n--- All Chart is saved. ---")

if not final_df.dropna(subset=['UniqueID', '시도명']).empty:
    save_top100_plots_by_sido(final_df)