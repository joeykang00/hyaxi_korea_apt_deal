import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import seaborn as sns

import matplotlib


# 1) Data: 기준금리
# 1. 데이터 불러오기
rate_csv = pd.read_csv('/home/kanghun/수업_AXI/기말고사/1_Korean_Apartment_Deal/ECOS_기준금리.csv', low_memory=False)# 첫행 기준금리만 선택
rate_row = rate_csv.iloc[0]
# 날짜만 추출
rate_row = rate_row[4:]
#행열 전환
rate_df = rate_row.reset_index()
rate_df.columns = ['date', 'rate']
#타입 변환
rate_df['date'] = pd.to_datetime(rate_df['date'])
rate_df['rate'] = pd.to_numeric(rate_df['rate'], errors='coerce')
rate_df = rate_df.rename(columns={'date':'날짜',
                                  'rate':'기준금리'})

# 2) Data: 인구수
# 1. 데이터 불러오기 : 첫 번째 줄: 날짜, 두 번째 줄: 항목명 (멀티컬럼 구조)
population_csv = pd.read_csv('/home/kanghun/수업_AXI/기말고사/1_Korean_Apartment_Deal/KOSIS_행정구역별_성별_인구수.csv', encoding="cp949", header=[0, 1])
# 2) 헤더 정리
population_csv.columns = pd.MultiIndex.from_tuples([(str(a).strip(), str(b).strip()) for a,b in population_csv.columns])

# 3) 첫 열 = 행정구역 시리즈 확보
region = population_csv.iloc[:, 0].astype(str).str.strip()   # 예: 전국, 서울특별시, ...

# 4) 나머지 중 '총인구수'만 선택
data = population_csv.iloc[:, 1:]  # 멀티헤더 구간
mask_total = data.columns.get_level_values(1).str.contains("총인구수")
total_only = data.loc[:, mask_total]

# 5) 행정구역을 행 인덱스로 지정
total_only.index = region

# 6) 열의 1레벨(날짜)만 사용하고 Datetime으로 변환
dates = total_only.columns.get_level_values(0)
total_only.columns = pd.to_datetime(dates, format="%Y.%m", errors="coerce")

# 7) 전치 후 날짜를 첫 컬럼으로
population_df = total_only.T.reset_index().rename(columns={"index": "날짜"})
population_df.columns.name = "행정구역"
population_df.drop(columns=['전국'], inplace=True)



# 3) Data: 실업률
import re
# 1. 데이터 불러오기 : 첫 번째 줄: 날짜, 두 번째 줄: 항목명 (멀티컬럼 구조)
unemployment_csv = pd.read_csv('/home/kanghun/수업_AXI/기말고사/1_Korean_Apartment_Deal/KOSIS_행정구역별_성별_실업률.csv', encoding="cp949", header=[0, 1])
# 2) 헤더 정리
unemployment_csv.columns = pd.MultiIndex.from_tuples([(str(a).strip(), str(b).strip()) for a,b in unemployment_csv.columns])
# 3) 첫 두 컬럼을 위치로 고정: [시도/행정구역], [성별]
region_col, gender_col = unemployment_csv.columns[0], unemployment_csv.columns[1]
region = unemployment_csv[region_col].astype(str).str.strip()
gender = unemployment_csv[gender_col].astype(str).str.strip()

# 4) 행 필터: 성별 == '계' 만 사용, '계' -> '전국'
row_mask = (gender == '계')
region = region[row_mask].replace({'계': '전국'})
df_rows = unemployment_csv.loc[row_mask]

# 5) 열 필터: 1레벨이 날짜(YYYY.MM) 모양인 것만
date_pat = re.compile(r'^\d{4}\.\d{2}$')
col_mask = [bool(date_pat.match(a)) for a in df_rows.columns.get_level_values(0)]
data = df_rows.loc[:, col_mask]

#    (중요) 날짜별로 2레벨(계/남/여)이 섞여 있을 수 있으므로,
#    1레벨(날짜) 기준으로 그룹화해 첫 열을 선택(= '계' 열이 보통 첫 번째)
# data = data.groupby(level=0, axis=1).first()
data = data.T.groupby(level=0).first().T

# 6) 행정구역을 인덱스로, 열(날짜)을 datetime으로
data.index = region.values
data.columns = pd.to_datetime(data.columns, format="%Y.%m")

# 7) 전치해서 "날짜"를 첫 컬럼으로
unemployment_df = data.T.reset_index().rename(columns={"index": "날짜"})

# 8) 숫자형 보정(혹시 문자열 섞여 있으면)
for c in unemployment_df.columns[1:]:
    unemployment_df[c] = pd.to_numeric(unemployment_df[c], errors="coerce")

# 2017이전 세종시 실업률 0으로 채우기
unemployment_df['세종특별자치시'] = unemployment_df['세종특별자치시'].fillna(0)



# 4) Data: 소비자물가지수
# 1. 데이터 불러오기 : 첫 번째 줄: 날짜, 두 번째 줄: 항목명 (멀티컬럼 구조)
price_csv = pd.read_csv('/home/kanghun/수업_AXI/기말고사/1_Korean_Apartment_Deal/ECOS_소비자물가지수.csv', encoding="UTF-8-SIG")
# 0행, 1행만 선택
subset = price_csv.iloc[[0, 1], 5:]   # 5번째 열 이후만 (즉 날짜 컬럼들만)
subset.index = ["총지수", "전년동기대비증감률"]  # 행 이름 부여

# melt로 세로 변환
cunsumer_price_df = subset.T.reset_index()
cunsumer_price_df.columns = ["날짜", "총지수", "전년동기대비증감률"]

# 숫자형 변환 (문자일 경우)
cunsumer_price_df["총지수"] = pd.to_numeric(cunsumer_price_df["총지수"], errors="coerce")
cunsumer_price_df["전년동기대비증감률"] = pd.to_numeric(cunsumer_price_df["전년동기대비증감률"], errors="coerce")

# 날짜를 datetime으로 바꾸기
cunsumer_price_df["날짜"] = pd.to_datetime(cunsumer_price_df["날짜"].astype(str), errors="coerce", format="%Y/%m")
cunsumer_price_df = cunsumer_price_df.rename(columns={"총지수" : "CPI_총지수",
                                                      "전년동기대비증감률" : "CPI_전년동기"})


#5) Data: 가계대출금
import re
# 1. 데이터 불러오기 : 첫 번째 줄: 날짜, 두 번째 줄: 항목명 (멀티컬럼 구조)
household_loan_csv = pd.read_csv('/home/kanghun/수업_AXI/기말고사/1_Korean_Apartment_Deal/ECOS_예금은행_지역별_대출금_말잔.csv', encoding="EUC-KR")

# 1️⃣ 계정항목이 '원화대출금'인 행만 필터링
household_loan_csv = household_loan_csv[household_loan_csv["계정항목별"].str.contains("원화대출금", na=False)]

# 2️⃣ 날짜 컬럼만 추출 (4번째 열 이후가 날짜)
date_cols = household_loan_csv.columns[3:]

# 3️⃣ melt로 세로 변환 후 pivot
melted = household_loan_csv.melt(id_vars=["지역코드별"], value_vars=date_cols,
                 var_name="날짜", value_name="대출금액")

# 4) 날짜 변환: "YYYY.MM" -> datetime 월초
melted["날짜"] = pd.to_datetime(melted["날짜"].astype(str),
                               format="%Y.%m", errors="coerce") + pd.offsets.MonthBegin(0)

# (원한다면 보기용 문자열로)
# melted["날짜"] = melted["날짜"].dt.strftime("%Y-%m-%d")

# 5) pivot: 1열=날짜, 2열~=지역
household_loan_df = (melted.pivot(index="날짜", columns="지역코드별", values="대출금액")
                    .sort_index()
                    .reset_index())

# 6) 숫자형 보정(안전)
for c in household_loan_df.columns[1:]:
    household_loan_df[c] = pd.to_numeric(household_loan_df[c], errors="coerce")


#6) Data: 개인소득
import re
import numpy as np
# 1. 데이터 불러오기 : 첫 번째 줄: 날짜, 두 번째 줄: 항목명 (멀티컬럼 구조)
income_csv = pd.read_csv('/home/kanghun/수업_AXI/기말고사/1_Korean_Apartment_Deal/KOSIS_시도별_개인소득.csv', encoding="EUC-KR", header=[0,1])

# 멀티컬럼 정리 (공백 제거)
income_csv.columns = pd.MultiIndex.from_tuples(
    [(str(a).strip(), str(b).strip()) for a,b in income_csv.columns]
)

# 지역 이름
regions = income_csv[("시도별", "시도별")].astype(str).str.strip()

# 3) '1인당 개인소득' 컬럼들의 연도 목록 추출
personal_cols = [c for c in income_csv.columns if c[1] == "1인당 개인소득"]
# 4) 연도 기준으로 정렬 (연도 문자열에서 숫자만 뽑아서 정렬)
def extract_year(col):
    return int(re.sub(r"[^0-9]", "", col[0]))

personal_cols = sorted(personal_cols, key=extract_year)

monthly_list = []

for col in personal_cols:
    raw_year = col[0]                         # 예: '2014', '2023 p)'
    year = int(re.sub(r"[^0-9]", "", raw_year))
    
    # ✅ 이게 핵심: 컬럼 접근은 그대로 `income_csv[col]` 사용
    annual_income = income_csv[col].astype(float)
    
    # 해당 연도의 월초 날짜 12개 생성
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
    
    # 결과 DF 뼈대
    monthly = pd.DataFrame({"날짜": dates})
    
    # 시도별로 연소득 / 12 값 채우기 (단위 : 만원)
    for region, value in zip(regions, annual_income):
        monthly[region] = np.round(float(value) / 12 / 10, 1)

    monthly_list.append(monthly)

# 5) 2014~2023 모든 연도 월별 소득 합치기
income_df = pd.concat(monthly_list, ignore_index=True)



# 기준 열 이름 리스트
columns_standard = [ '날짜', '강원', '경기', '경남', '경북', '광주', '대구', '대전', '부산', '서울', 
                     '세종', '울산', '인천', '전남', '전북', '제주', '충남', '충북' ]

# 각 데이터프레임의 열 이름을 표준화 (이름 통일)
rename_map = {'울산광역시'         : '울산',
              '세종특별자치시'      : '세종',
              '경기도'             : '경기',
              '강원도'             : '강원',
              '강원특별자치도'      :'강원',
              '충청북도'           : '충북',
              '충청남도'           : '충남',
              '전라북도'           : '전북',
              '전북특별자치도'      : '전북',             
              '전라남도'           : '전남',
              '경상북도'           : '경북',
              '경상남도'           : '경남',
              '제주특별자치도'      : '제주',
              '제주도'             : '제주',
              '서울특별시'         : '서울',
              '부산광역시'         : '부산',
              '대구광역시'         : '대구',
              '인천광역시'         : '인천',
              '광주광역시'         : '광주',
              '대전광역시'         : '대전',
              '전국'              : None,  # 필요 없으면 제거
              '거래일'             : '날짜',
            }


# rename 후 '전국' 열 제거 및 순서 재정렬
def standardize_columns(df):
    df = df.rename(columns=rename_map).drop(columns=[None], errors='ignore')
    df = df.reindex(columns=columns_standard)
    return df

df1 = household_loan_df
df2 = population_df
df3 = unemployment_df

df4 = rate_df
df5 = cunsumer_price_df

df6 = income_df


# 세 데이터프레임에 적용
df1 = standardize_columns(df1)
df2 = standardize_columns(df2)
df3 = standardize_columns(df3)
df6 = standardize_columns(df6)


# 날짜형식 통일

df1['날짜'] = pd.to_datetime(df1['날짜'], errors='coerce')
df2['날짜'] = pd.to_datetime(df2['날짜'], errors='coerce')
df3['날짜'] = pd.to_datetime(df3['날짜'], errors='coerce')
df4['날짜'] = pd.to_datetime(df4['날짜'], errors='coerce')
df5['날짜'] = pd.to_datetime(df5['날짜'], errors='coerce')
df6['날짜'] = pd.to_datetime(df6['날짜'], errors='coerce')


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

data_dir = '/home/kanghun/수업_AXI/기말고사/1_Korean_Apartment_Deal'
deal_csv_path = prepare_csv_from_zip(data_dir, 'KoreaApartDeal.csv', 'KoreaApartDeal.zip')
loc_csv_path = prepare_csv_from_zip(data_dir, 'LocationCode.csv', 'LocationCode.zip')

try:
    deal_df = pd.read_csv(deal_csv_path)
    location_df = pd.read_csv(loc_csv_path, dtype={'법정동코드': str, '읍면동명': str, '리명': str})
except Exception as e:
    print(f"file read error: {e}")
    exit()
    
print(f"\n# of Total Raw Data: {len(deal_df):,}")
initial_rows = len(deal_df)
deal_df.dropna(subset=['거래일', '거래금액'], inplace=True)
final_rows = len(deal_df)

# pd.set_option('display.max_rows', None)       # 행 제한 해제
pd.set_option('display.max_columns', None)    # 열 제한 해제
pd.set_option('display.max_colwidth', None)   # 컬럼 내용 길이 제한 해제


# --- Data Merging for '시군구명' ---
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


# --- Data Cleaning for '거래금액' ---
print("\n--- Cleaning and standardizing the '거래금액' column ---")
final_df['거래금액'] = pd.to_numeric(final_df['거래금액'].astype(str).str.replace(',', ''), errors='coerce')

# --- Data Cleaning for '거래일' Column ---
print("\n--- Cleaning and standardizing the '거래일' column ---")
final_df['거래일_정리'] = final_df['거래일'].astype(str).str.split(' ').str[0]
final_df['거래일_정리'] = pd.to_datetime(final_df['거래일_정리'], format='mixed', errors='coerce')
invalid_dates = final_df[final_df['거래일_정리'].isnull()]
if not invalid_dates.empty:
    print("\n[Warning] The following rows could not be converted to a valid date:")
    print(invalid_dates[['거래일']]) # 문제가 된 원본 '거래일' 데이터 출력
final_df.dropna(subset=['거래일_정리'], inplace=True)
final_df['거래일'] = final_df['거래일_정리'].dt.date
final_df.drop(columns=['거래일_정리'], inplace=True)

final_columns = [
    'UniqueID', '시도명', '시군구명', '법정동', '아파트',
    '전용면적', '거래일', '거래금액', '층', '건축년도'
]
final_columns_exist = [col for col in final_columns if col in final_df.columns]
final_df = final_df[final_columns_exist]
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
print("\n'UniqueID' and final:")
print(final_df)


apt_price_df = final_df.copy()
# 시도명 표준화
apt_price_df['시도명'] = apt_price_df['시도명'].replace(rename_map)
apt_price_df = apt_price_df.dropna(subset=['시도명'])

# 날짜형식 통일
apt_price_df['거래일'] = pd.to_datetime(apt_price_df['거래일'], errors='coerce')
print(apt_price_df['거래일'].dtype)

apt_price_df['건축년도'] = apt_price_df['건축년도'].fillna(2021)


apt_price_df = pd.merge_asof(
    apt_price_df.sort_values('거래일'),
    df4.sort_values('날짜'),
    left_on='거래일',
    right_on='날짜',
    direction='forward'
)
apt_price_df.drop(columns='날짜', inplace=True)

# 거래일을 월 초로 맞추기 (인구 데이터는 월별 기준이므로)
apt_price_df["월기준"] = apt_price_df["거래일"].values.astype("datetime64[M]")
print(apt_price_df['거래일'].dtype)

# 가계대출금 합치기
# 지역별 대출금 wide → long
loan_long = df1.melt(
    id_vars=["날짜"],          # 날짜는 그대로 두고
    var_name="시도명",        # 열 이름 → 시도명
    value_name="가계대출(만원)"       # 값 → 대출금
)
apt_price_df = pd.merge(
    apt_price_df,
    loan_long,
    left_on=["월기준", "시도명"],
    right_on=["날짜", "시도명"],
    how="left"
).drop(columns=["날짜"])


#인구수 데이터 합치기
# # 4 인구 데이터 melt (wide → long 형태로 변환: 시도명 / 날짜 / 인구)
pop_long = df2.melt(
    id_vars=["날짜"],
    var_name="시도명",
    value_name="인구수"
)
apt_price_df = pd.merge(
    apt_price_df,
    pop_long,
    left_on=["월기준", "시도명"],
    right_on=["날짜", "시도명"],
    how="left"
).drop(columns=["날짜"])

# 실업률 데이터 합치기
# 지역별 실업률 wide → long
unemp_long = df3.melt(
    id_vars=["날짜"],
    var_name="시도명",
    value_name="실업률"
)

apt_price_df = pd.merge(
    apt_price_df,
    unemp_long,
    left_on=["월기준", "시도명"],
    right_on=["날짜", "시도명"],
    how="left"
).drop(columns=["날짜"])

# 소비자물가지수 합치기
df5["월기준"] = df5["날짜"].values.astype("datetime64[M]")
apt_price_df = pd.merge(
    apt_price_df,
    df5,
    on="월기준",
    how="left"
).drop(columns=["날짜"])

# 개인소득 합치기
income_long = df6.melt(
    id_vars=["날짜"],
    var_name="시도명",
    value_name="월개인소득(만원)"
)

apt_price_df = pd.merge(
    apt_price_df,
    income_long,
    left_on=["월기준", "시도명"],
    right_on=["날짜", "시도명"],
    how="left"
).drop(columns=["날짜"])
