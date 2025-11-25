import os
import re
import zipfile
from glob import glob

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import seaborn as sns


# ======================================================================
# 기본 설정
# ======================================================================

DATA_DIR = "/home/kanghun/수업_AXI/기말고사/1_Korean_Apartment_Deal"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)


# ======================================================================
# 공통 유틸 함수
# ======================================================================

def prepare_csv_from_zip(data_dir: str, csv_filename: str, zip_filename: str) -> str:
    """
    data_dir 안에서 csv_filename 이 없으면 zip_filename 을 풀어서 생성해 주는 함수.
    최종적으로 csv 파일의 전체 경로를 반환.
    """
    csv_path = os.path.join(data_dir, csv_filename)
    zip_path = os.path.join(data_dir, zip_filename)

    if not os.path.exists(csv_path):
        print(f"'{csv_path}' is not exist. Checking Zip file.")
        if os.path.exists(zip_path):
            print(f"'{zip_path}' is found. Unzip...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                print(f"Completed Unzip. '{csv_path}' will be used.")
            except Exception as e:
                print(f"Unzip error: {e}")
                exit()
        else:
            print(f"Error: '{csv_filename}' and '{zip_filename}' are not found.")
            exit()
    return csv_path


# 표준 열 이름
COLUMNS_STANDARD = [
    "날짜", "강원", "경기", "경남", "경북", "광주", "대구", "대전",
    "부산", "서울", "세종", "울산", "인천", "전남", "전북", "제주",
    "충남", "충북",
]

# 시도명 표준화 매핑
RENAME_MAP = {
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
    "제주도": "제주",
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "전국": None,      # 필요 없으면 제거
    "거래일": "날짜",   # 일부 데이터에서 '거래일' → '날짜'
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    시도명/날짜 열 이름을 RENAME_MAP 과 COLUMNS_STANDARD 기준으로 맞춰주는 함수.
    """
    df = df.rename(columns=RENAME_MAP).drop(columns=[None], errors="ignore")
    df = df.reindex(columns=COLUMNS_STANDARD)
    return df

def find_file_by_pattern(data_dir: str, pattern: str) -> str:
    """
    data_dir 안에서 pattern(*.csv 등)에 매칭되는 파일을 찾아서
    하나만 있으면 그 경로를 반환.
    없거나 여러 개면 에러 메시지 출력 후 종료.
    """
    search_pattern = os.path.join(data_dir, pattern)
    matches = glob(search_pattern)

    if len(matches) == 0:
        print(f"[Error] No file matched pattern: {search_pattern}")
        exit()
    elif len(matches) > 1:
        print(f"[Error] Multiple files matched pattern: {search_pattern}")
        for m in matches:
            print(" -", m)
        print("패턴에 하나만 매칭되도록 파일명을 정리해 주세요.")
        exit()

    return matches[0]

# ======================================================================
# 1) 기준금리 데이터
# ======================================================================

def load_base_rate(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_기준금리.csv")
    path = find_file_by_pattern(data_dir, "*_기준금리.csv")
    rate_csv = pd.read_csv(path, low_memory=False)

    rate_row = rate_csv.iloc[0, 4:]  # 첫 행에서 5번째 열부터 날짜 구간만
    rate_df = rate_row.reset_index()
    rate_df.columns = ["date", "rate"]

    rate_df["date"] = pd.to_datetime(rate_df["date"])
    rate_df["rate"] = pd.to_numeric(rate_df["rate"], errors="coerce")

    rate_df = rate_df.rename(columns={"date": "날짜", "rate": "기준금리"})
    return rate_df


# ======================================================================
# 2) 인구수 데이터
# ======================================================================

def load_population(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_인구수.csv")
    path = find_file_by_pattern(data_dir, "*_인구수.csv")

    population_csv = pd.read_csv(
        path,
        encoding="cp949",
        header=[0, 1],
    )

    # 멀티컬럼 정리
    population_csv.columns = pd.MultiIndex.from_tuples(
        [(str(a).strip(), str(b).strip()) for a, b in population_csv.columns]
    )

    # 첫 열: 행정구역
    region = population_csv.iloc[:, 0].astype(str).str.strip()

    # '총인구수' 컬럼만 선택
    data = population_csv.iloc[:, 1:]
    mask_total = data.columns.get_level_values(1).str.contains("총인구수")
    total_only = data.loc[:, mask_total]

    # 행정구역을 인덱스로
    total_only.index = region

    # 열의 1레벨(날짜)만 사용 → datetime
    dates = total_only.columns.get_level_values(0)
    total_only.columns = pd.to_datetime(dates, format="%Y.%m", errors="coerce")

    # 전치 후 날짜를 첫 컬럼으로
    population_df = total_only.T.reset_index().rename(columns={"index": "날짜"})
    population_df.columns.name = "행정구역"

    # 전국 제거
    population_df.drop(columns=["전국"], inplace=True)
    return population_df


# ======================================================================
# 3) 실업률 데이터
# ======================================================================

def load_unemployment(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_실업률.csv")
    path = find_file_by_pattern(data_dir, "*_실업률.csv")

    unemployment_csv = pd.read_csv(
        path,
        encoding="cp949",
        header=[0, 1],
    )

    # 멀티헤더 정리
    unemployment_csv.columns = pd.MultiIndex.from_tuples(
        [(str(a).strip(), str(b).strip()) for a, b in unemployment_csv.columns]
    )

    # 첫 두 컬럼: [시도/행정구역], [성별]
    region_col, gender_col = unemployment_csv.columns[0], unemployment_csv.columns[1]
    region = unemployment_csv[region_col].astype(str).str.strip()
    gender = unemployment_csv[gender_col].astype(str).str.strip()

    # 성별 == '계' 만 사용
    row_mask = gender == "계"
    region = region[row_mask].replace({"계": "전국"})
    df_rows = unemployment_csv.loc[row_mask]

    # 날짜 패턴 필터 (YYYY.MM)
    date_pat = re.compile(r"^\d{4}\.\d{2}$")
    col_mask = [
        bool(date_pat.match(a)) for a in df_rows.columns.get_level_values(0)
    ]
    data = df_rows.loc[:, col_mask]

    # 1레벨(날짜) 기준 그룹 → 첫 컬럼(보통 '계') 사용
    data = data.T.groupby(level=0).first().T

    # 행정구역을 인덱스로, 열은 datetime
    data.index = region.values
    data.columns = pd.to_datetime(data.columns, format="%Y.%m")

    # 전치해서 날짜를 첫 컬럼으로
    unemployment_df = data.T.reset_index().rename(columns={"index": "날짜"})

    # 숫자형 변환
    for c in unemployment_df.columns[1:]:
        unemployment_df[c] = pd.to_numeric(unemployment_df[c], errors="coerce")

    # 2017 이전 세종시 실업률 0으로 채우기
    if "세종특별자치시" in unemployment_df.columns:
        unemployment_df["세종특별자치시"] = unemployment_df["세종특별자치시"].fillna(0)

    return unemployment_df


# ======================================================================
# 4) 소비자물가지수(CPI)
# ======================================================================

def load_cpi(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_소비자물가지수.csv")
    path = find_file_by_pattern(data_dir, "*_소비자물가지수.csv")
    price_csv = pd.read_csv(path, encoding="UTF-8-SIG")

    # 0행, 1행 + 5번째 열 이후 (날짜 컬럼들만)
    subset = price_csv.iloc[[0, 1], 5:]
    subset.index = ["총지수", "전년동기대비증감률"]

    cpi_df = subset.T.reset_index()
    cpi_df.columns = ["날짜", "총지수", "전년동기대비증감률"]

    cpi_df["총지수"] = pd.to_numeric(cpi_df["총지수"], errors="coerce")
    cpi_df["전년동기대비증감률"] = pd.to_numeric(
        cpi_df["전년동기대비증감률"], errors="coerce"
    )

    cpi_df["날짜"] = pd.to_datetime(
        cpi_df["날짜"].astype(str),
        errors="coerce",
        format="%Y/%m",
    )

    cpi_df = cpi_df.rename(
        columns={
            "총지수": "CPI_총지수",
            "전년동기대비증감률": "CPI_전년동기",
        }
    )
    return cpi_df


# ======================================================================
# 5) 가계대출금 데이터
# ======================================================================

def load_household_loan(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_대출금(말잔).csv")
    path = find_file_by_pattern(data_dir, "*_가계대출.csv")

    df = pd.read_csv(path, encoding="UTF-8-SIG")

    # 계정항목이 '원화대출금' 인 행만
    df = df[df["계정항목"].str.contains("원화대출금", na=False)]

    date_cols = df.columns[5:]

    melted = df.melt(
        id_vars=["지역코드"],
        value_vars=date_cols,
        var_name="날짜",
        value_name="대출금액",
    )

    melted["날짜"] = pd.to_datetime(
        melted["날짜"].astype(str),
        format="%Y/%m",
        errors="coerce",
    ) + pd.offsets.MonthBegin(0)


    # 6. 숫자형 변환
    melted["대출금액"] = pd.to_numeric(melted["대출금액"].astype(str).str.replace(",", ""),
                                errors="coerce")
    
    household_loan_df = (
        melted.pivot(index="날짜", columns="지역코드", values="대출금액")
        .sort_index()
        .reset_index()
    )

    for c in household_loan_df.columns[1:]:
        household_loan_df[c] = pd.to_numeric(household_loan_df[c], errors="coerce")

    return household_loan_df

# ======================================================================
# 6) 은헹대출 데이터
# ======================================================================

def load_bank_loan(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_대출금(말잔).csv")
    path = find_file_by_pattern(data_dir, "*_은행대출.csv")

    df = pd.read_csv(path, encoding="EUC-KR")

    # 계정항목이 '원화대출금' 인 행만
    df = df[df["계정항목별"].str.contains("원화대출금", na=False)]

    date_cols = df.columns[3:]

    melted = df.melt(
        id_vars=["지역코드별"],
        value_vars=date_cols,
        var_name="날짜",
        value_name="대출금액",
    )

    melted["날짜"] = pd.to_datetime(
        melted["날짜"].astype(str),
        format="%Y.%m",
        errors="coerce",
    ) + pd.offsets.MonthBegin(0)

    bank_loan_df = (
        melted.pivot(index="날짜", columns="지역코드별", values="대출금액")
        .sort_index()
        .reset_index()
    )

    for c in bank_loan_df.columns[1:]:
        bank_loan_df[c] = pd.to_numeric(bank_loan_df[c], errors="coerce")

    return bank_loan_df

# ======================================================================
# 7) 개인소득 데이터
# ======================================================================

def load_income(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_개인소득.csv")
    path = find_file_by_pattern(data_dir, "*_개인소득.csv")

    income_csv = pd.read_csv(path, encoding="EUC-KR", header=[0, 1])

    # 멀티컬럼 정리
    income_csv.columns = pd.MultiIndex.from_tuples(
        [(str(a).strip(), str(b).strip()) for a, b in income_csv.columns]
    )

    regions = income_csv[("시도별", "시도별")].astype(str).str.strip()

    # '1인당 개인소득' 컬럼만 사용
    personal_cols = [
        c for c in income_csv.columns if c[1] == "1인당 개인소득"
    ]

    def extract_year(col):
        return int(re.sub(r"[^0-9]", "", col[0]))

    personal_cols = sorted(personal_cols, key=extract_year)

    monthly_list = []

    for col in personal_cols:
        raw_year = col[0]
        year = int(re.sub(r"[^0-9]", "", raw_year))

        annual_income = income_csv[col].astype(float)

        # 해당 연도의 월초 날짜 12개
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
        monthly = pd.DataFrame({"날짜": dates})

        # 시도별로 연소득 / 12 / 10 (단위: 만원)
        for region, value in zip(regions, annual_income):
            monthly[region] = np.round(float(value) / 12 / 10, 1)

        monthly_list.append(monthly)

    income_df = pd.concat(monthly_list, ignore_index=True)
    return income_df


# ======================================================================
# 8) 아파트 실거래 + 위치코드 데이터 로드
# ======================================================================

def load_apartment_deal_and_location(data_dir: str):
    deal_csv_path = prepare_csv_from_zip(data_dir, "KoreaApartDeal.csv", "KoreaApartDeal.zip")
    loc_csv_path = prepare_csv_from_zip(data_dir, "LocationCode.csv", "LocationCode.zip")

    try:
        deal_df = pd.read_csv(deal_csv_path)
        location_df = pd.read_csv(
            loc_csv_path,
            dtype={"법정동코드": str, "읍면동명": str, "리명": str},
        )
    except Exception as e:
        print(f"file read error: {e}")
        exit()

    print(f"\n# of Total Raw Data: {len(deal_df):,}")

    # 거래일/거래금액 결측 제거
    initial_rows = len(deal_df)
    deal_df.dropna(subset=["거래일", "거래금액"], inplace=True)
    final_rows = len(deal_df)

    # 시군구명 매핑
    location_df_filtered = location_df[location_df["시군구명"].notna()].copy()
    location_df_filtered["지역코드"] = location_df_filtered["법정동코드"].str[:5].astype(int)
    loc_map = location_df_filtered[["지역코드", "시도명", "시군구명"]].drop_duplicates()

    df = pd.merge(deal_df, loc_map, on="지역코드", how="left")

    # 법정동 코드 매핑
    loc_lookup = location_df[["시도명", "시군구명", "읍면동명", "리명", "법정동코드"]].copy()
    loc_lookup["법정동"] = (
        loc_lookup["읍면동명"].fillna("") + " " + loc_lookup["리명"].fillna("")
    ).str.strip()

    final_df = pd.merge(
        df,
        loc_lookup[["시도명", "시군구명", "법정동", "법정동코드"]],
        on=["시도명", "시군구명", "법정동"],
        how="left",
    )

    # UniqueID 구성
    final_df["아파트ID"] = pd.factorize(final_df["아파트"])[0]
    final_df["아파트ID"] = final_df["아파트ID"].astype(str).str.zfill(5)
    final_df["전용면적ID"] = final_df["전용면적"].round(0).astype(int).astype(str).str.zfill(3)
    final_df["UniqueID"] = final_df["법정동코드"] + final_df["아파트ID"] + final_df["전용면적ID"]

    # 거래금액 정리
    print("\n--- Cleaning and standardizing the '거래금액' column ---")
    final_df["거래금액"] = pd.to_numeric(
        final_df["거래금액"].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    # 거래일 정리
    print("\n--- Cleaning and standardizing the '거래일' column ---")
    final_df["거래일_정리"] = final_df["거래일"].astype(str).str.split(" ").str[0]
    final_df["거래일_정리"] = pd.to_datetime(
        final_df["거래일_정리"],
        format="mixed",
        errors="coerce",
    )

    invalid_dates = final_df[final_df["거래일_정리"].isnull()]
    if not invalid_dates.empty:
        print("\n[Warning] The following rows could not be converted to a valid date:")
        print(invalid_dates[["거래일"]])

    final_df.dropna(subset=["거래일_정리"], inplace=True)
    final_df["거래일"] = final_df["거래일_정리"].dt.date
    final_df.drop(columns=["거래일_정리"], inplace=True)

    final_columns = [
        "UniqueID", "시도명", "시군구명", "법정동", "아파트",
        "전용면적", "거래일", "거래금액", "층", "건축년도",
    ]
    final_columns_exist = [col for col in final_columns if col in final_df.columns]
    final_df = final_df[final_columns_exist]

    print("\n'UniqueID' and final:")
    print(final_df)

    return final_df


# ======================================================================
# 8) 전체 머지 파이프라인
# ======================================================================

def build_apt_price_with_macro():
    # 매크로 데이터 로드
    rate_df = load_base_rate(DATA_DIR)
    population_df = load_population(DATA_DIR)
    unemployment_df = load_unemployment(DATA_DIR)
    cpi_df = load_cpi(DATA_DIR)
    household_loan_df = load_household_loan(DATA_DIR)
    bank_loan_df = load_bank_loan(DATA_DIR)
    income_df = load_income(DATA_DIR)

    # 시도명/열 표준화
    df1 = standardize_columns(household_loan_df)
    df2 = standardize_columns(population_df)
    df3 = standardize_columns(unemployment_df)
    df6 = standardize_columns(bank_loan_df)
    df7 = standardize_columns(income_df)

    # 날짜 형식 통일
    for df_tmp in [df1, df2, df3, rate_df, cpi_df, df6, df7]:
        df_tmp["날짜"] = pd.to_datetime(df_tmp["날짜"], errors="coerce")

    # 아파트 실거래/위치 데이터
    final_df = load_apartment_deal_and_location(DATA_DIR)

    # 아파트 가격 DF
    apt_price_df = final_df.copy()

    # 시도명 표준화 (값 기준)
    apt_price_df["시도명"] = apt_price_df["시도명"].replace(RENAME_MAP)
    apt_price_df = apt_price_df.dropna(subset=["시도명"])

    # 거래일 datetime 변환
    apt_price_df["거래일"] = pd.to_datetime(apt_price_df["거래일"], errors="coerce")
    print(apt_price_df["거래일"].dtype)

    # 건축년도 결측 채우기
    apt_price_df["건축년도"] = apt_price_df["건축년도"].fillna(2021)

    # 기준금리 asof 머지
    apt_price_df = pd.merge_asof(
        apt_price_df.sort_values("거래일"),
        rate_df.sort_values("날짜"),
        left_on="거래일",
        right_on="날짜",
        direction="forward",
    ).drop(columns=["날짜"])

    # 거래일을 월초 기준으로 (월 단위 매크로와 align)
    apt_price_df["월기준"] = apt_price_df["거래일"].values.astype("datetime64[M]")
    print(apt_price_df["거래일"].dtype)

    # -------------------------
    # 가계대출금 합치기
    # -------------------------
    loan_long = df1.melt(
        id_vars=["날짜"],
        var_name="시도명",
        value_name="가계대출(만원)",
    )

    apt_price_df = pd.merge(
        apt_price_df,
        loan_long,
        left_on=["월기준", "시도명"],
        right_on=["날짜", "시도명"],
        how="left",
    ).drop(columns=["날짜"])

        # -------------------------
    # 은행대출금 합치기
    # -------------------------
    loan_long = df6.melt(
        id_vars=["날짜"],
        var_name="시도명",
        value_name="가계대출(만원)",
    )

    apt_price_df = pd.merge(
        apt_price_df,
        loan_long,
        left_on=["월기준", "시도명"],
        right_on=["날짜", "시도명"],
        how="left",
    ).drop(columns=["날짜"])

    # -------------------------
    # 인구수 합치기
    # -------------------------
    pop_long = df2.melt(
        id_vars=["날짜"],
        var_name="시도명",
        value_name="인구수",
    )

    apt_price_df = pd.merge(
        apt_price_df,
        pop_long,
        left_on=["월기준", "시도명"],
        right_on=["날짜", "시도명"],
        how="left",
    ).drop(columns=["날짜"])

    # -------------------------
    # 실업률 합치기
    # -------------------------
    unemp_long = df3.melt(
        id_vars=["날짜"],
        var_name="시도명",
        value_name="실업률",
    )

    apt_price_df = pd.merge(
        apt_price_df,
        unemp_long,
        left_on=["월기준", "시도명"],
        right_on=["날짜", "시도명"],
        how="left",
    ).drop(columns=["날짜"])

    # -------------------------
    # CPI (소비자물가지수) 합치기
    # -------------------------
    cpi_df["월기준"] = cpi_df["날짜"].values.astype("datetime64[M]")

    apt_price_df = pd.merge(
        apt_price_df,
        cpi_df,
        on="월기준",
        how="left",
    ).drop(columns=["날짜"])

    # -------------------------
    # 개인소득 합치기
    # -------------------------
    income_long = df7.melt(
        id_vars=["날짜"],
        var_name="시도명",
        value_name="월개인소득(만원)",
    )

    apt_price_df = pd.merge(
        apt_price_df,
        income_long,
        left_on=["월기준", "시도명"],
        right_on=["날짜", "시도명"],
        how="left",
    ).drop(columns=["날짜"])

    return apt_price_df


# ======================================================================
# 메인 실행
# ======================================================================

if __name__ == "__main__":
    apt_price_df = build_apt_price_with_macro()
    print("\n[완료] 매크로 변수까지 합쳐진 아파트 거래 데이터:")
    print(apt_price_df.head())
