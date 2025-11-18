import pandas as pd
import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import seaborn as sns

import matplotlib


# 기준금리
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

# 인구수
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
