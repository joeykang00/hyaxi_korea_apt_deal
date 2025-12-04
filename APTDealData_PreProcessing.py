# pip install pandas matplotlib glob PIL

import os
import re
import platform
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm

from glob import glob
from PIL import Image


# ======================================================================
# 기본 설정
# ======================================================================

DATA_DIR = "./data/"
PREPROCESSED_DIR = "./preprocessed/"

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

def prepare_csv_from_zip(data_dir: str, csv_filename: str, zip_filename: str) -> str:
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


def create_split_zip(source_file, output_prefix, chunk_size_mb=100):

    temp_zip = "temp_zip_file.zip"

    print(f"creating temp zip file... ({temp_zip})")
    with zipfile.ZipFile(temp_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        zipf.write(source_file, arcname=os.path.basename(source_file))

    print(f"{chunk_size_mb:.1f}MB split zip...")

    part_num = 1
    with open(temp_zip, 'rb') as f:
        while True:
            chunk = f.read(chunk_size_mb*1024*1024)
            if not chunk:
                break

            part_filename = f"{output_prefix}.{part_num:03d}"

            with open(part_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)

            print(f"created   -> {part_filename}")
            part_num += 1

    # 3. 임시 파일 삭제
    os.remove(temp_zip)


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
    "전국": None,  # 필요 없으면 제거
    "거래일": "날짜",  # 일부 데이터에서 '거래일' → '날짜'
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


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame의 컬럼들을 가능한 작은 dtype으로 다운캐스팅해서
    메모리 사용량을 줄여줌.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        # 숫자형만 다운캐스트
        if str(col_type)[:3] == 'int':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif str(col_type)[:5] == 'float':
            df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Memory reduced: {start_mem:.3f} MB → {end_mem:.3f} MB")

    return df


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

    print("기준금리 data")
    print(rate_df)

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

    print("인구수 data")
    print(population_df)

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

    print("소비자물가지수 data")
    print(cpi_df)

    return cpi_df


# ======================================================================
# 5) 가계대출금 데이터
# ======================================================================

def load_household_loan(data_dir: str) -> pd.DataFrame:
    # 예: ECOS_가계대출.csv 같은 이름
    path = find_file_by_pattern(data_dir, "*_가계대출.csv")

    df = pd.read_csv(path, encoding="UTF-8-SIG")

    # '예금은행'만 사용
    df = df[df["계정항목"].str.contains("예금은행", na=False)]

    # 날짜 컬럼 (앞쪽 5개 컬럼 이후부터가 날짜)
    date_cols = df.columns[5:]

    melted = df.melt(
        id_vars=["지역코드"],
        value_vars=date_cols,
        var_name="날짜",
        value_name="대출금액",
    )

    # 날짜 변환: "YYYY/MM" → 월초 날짜
    melted["날짜"] = pd.to_datetime(
        melted["날짜"].astype(str),
        format="%Y/%m",
        errors="coerce",
    ) + pd.offsets.MonthBegin(0)

    # 숫자형 변환
    melted["대출금액"] = pd.to_numeric(
        melted["대출금액"].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    print("가계대출 data")
    print(melted)

    # 날짜 = index, 지역코드 = columns
    household_loan_df = (
        melted.pivot_table(
            index="날짜",
            columns="지역코드",
            values="대출금액",
            aggfunc="first",  # 중복 처리
        )
        .sort_index()
        .reset_index()
    )

    print("가계대출 pivot data")
    print(household_loan_df)

    return household_loan_df


# ======================================================================
# 6) 은헹대출 데이터
# ======================================================================

def load_bank_loan(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_대출금(말잔).csv")
    path = find_file_by_pattern(data_dir, "ECOS_예금은행_지역별_은행대출.csv")

    df = pd.read_csv(path, encoding="UTF-8-SIG", sep='\t')

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
        melted.pivot_table(index="날짜", columns="지역코드별", values="대출금액", aggfunc="first")
        .sort_index()
        .reset_index()
    )

    for c in bank_loan_df.columns[1:]:
        bank_loan_df[c] = pd.to_numeric(bank_loan_df[c], errors="coerce")

    print("은헹대출 data")
    print(bank_loan_df)

    return bank_loan_df


# ======================================================================
# 7) 개인소득 데이터
# ======================================================================

def load_income(data_dir: str) -> pd.DataFrame:
    # path = os.path.join(data_dir, "*_개인소득.csv")
    path = find_file_by_pattern(data_dir, "*_개인소득.csv")

    income_csv = pd.read_csv(path, encoding='cp949', header=[0, 1])

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

    print("개인소득 data")
    print(income_df)

    return income_df


# ======================================================================
# 8) 아파트 실거래 + 위치코드 데이터 로드
# ======================================================================

def load_apartment_deal_and_location(data_dir: str):
    deal_csv_path = prepare_csv_from_zip(data_dir, "KoreaApartDeal.csv", "KoreaApartDeal.zip")
    loc_csv_path = prepare_csv_from_zip(data_dir, "LocationCode.csv", "LocationCode.zip")

    try:
        deal_df = pd.read_csv(deal_csv_path, low_memory=False)
        location_df = pd.read_csv(
            loc_csv_path,
            dtype={"법정동코드": str, "읍면동명": str, "리명": str},
        )
    except Exception as e:
        print(f"file read error: {e}")
        exit()

    print(f"\n# of Total Apartment Deal Data: {len(deal_df):,}")

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

    # 🔥 다운캐스팅 적용
    final_df = reduce_memory_usage(final_df)

    print("\n'UniqueID' and apart deal data")
    print(final_df)

    return final_df


def build_apt_price_with_macro():
    # 매크로 데이터 로드
    rate_df = load_base_rate(DATA_DIR)
    population_df = load_population(DATA_DIR)
    unemployment_df = load_unemployment(DATA_DIR)
    cpi_df = load_cpi(DATA_DIR)
    household_loan_df = load_household_loan(DATA_DIR)
    bank_loan_df = load_bank_loan(DATA_DIR)
    income_df = load_income(DATA_DIR)

    print("\n--- Visualization Trend Charts ---")
    plot_macro_trends(population_df, unemployment_df, cpi_df, household_loan_df, bank_loan_df, rate_df, output_dir='preprocessed')


    # 🔥 다운캐스팅
    rate_df = reduce_memory_usage(rate_df)
    population_df = reduce_memory_usage(population_df)
    unemployment_df = reduce_memory_usage(unemployment_df)
    cpi_df = reduce_memory_usage(cpi_df)
    household_loan_df = reduce_memory_usage(household_loan_df)
    bank_loan_df = reduce_memory_usage(bank_loan_df)
    income_df = reduce_memory_usage(income_df)

    # 시도명/열 표준화
    std_household_loan_df = standardize_columns(household_loan_df)
    std_population_df = standardize_columns(population_df)
    std_unemployment_df = standardize_columns(unemployment_df)
    std_bank_loan_df = standardize_columns(bank_loan_df)
    std_income_df = standardize_columns(income_df)

    # 날짜 형식 통일
    for df_tmp in [std_household_loan_df, std_population_df, std_unemployment_df, rate_df, cpi_df, std_bank_loan_df, std_income_df]:
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
    # print(apt_price_df["거래일"].dtype)

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
    # print(apt_price_df["거래일"].dtype)

    # -------------------------
    # 가계대출금 합치기
    # -------------------------
    print("가계대출금 합치기 start")
    loan_long = std_household_loan_df.melt(
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
    print("가계대출금 합치기 end")

    # -------------------------
    # 은행대출금 합치기
    # -------------------------
    print("은행대출금 합치기 start")
    loan_long = std_bank_loan_df.melt(
        id_vars=["날짜"],
        var_name="시도명",
        value_name="은행대출(만원)",
    )

    apt_price_df = pd.merge(
        apt_price_df,
        loan_long,
        left_on=["월기준", "시도명"],
        right_on=["날짜", "시도명"],
        how="left",
    ).drop(columns=["날짜"])

    print("은행대출금 합치기 end")

    # -------------------------
    # 인구수 합치기
    # -------------------------
    print("인구수 합치기 start")
    pop_long = std_population_df.melt(
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
    print("인구수 합치기 end")

    # -------------------------
    # 실업률 합치기
    # -------------------------
    print("실업률 합치기 start")
    unemp_long = std_unemployment_df.melt(
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
    print("실업률 합치기 end")

    # -------------------------
    # CPI (소비자물가지수) 합치기
    # -------------------------
    print("소비자물가지수 합치기 start")
    cpi_df["월기준"] = cpi_df["날짜"].values.astype("datetime64[M]")

    apt_price_df = pd.merge(
        apt_price_df,
        cpi_df,
        on="월기준",
        how="left",
    ).drop(columns=["날짜"])
    print("소비자물가지수 합치기 end")

    # -------------------------
    # 개인소득 합치기
    # -------------------------
    print("개인소득 합치기 start")
    income_long = std_income_df.melt(
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
    print("개인소득 합치기 end")

    apt_price_df = apt_price_df.drop('월기준', axis=1)

    apt_price_df = reduce_memory_usage(apt_price_df)

    return apt_price_df

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
            print(f"'{font_family}' font is set for Windows.")
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
            print(f"'{font_family}' font is set for macOS.")
        else:
            print("Warning: Could not find a suitable Hangul font (Apple SD Gothic Neo, AppleGothic) on this macOS system. Characters may appear broken.")

    elif os_name == 'Linux':
        try:
            if fm.findfont('NanumGothic', fallback_to_default=False):
                plt.rc('font', family='NanumGothic')
                print("'NanumGothic' font is set for Linux.")
            else:
                print("Warning: 'NanumGothic' not found on this Linux system. Please install it. Characters may appear broken.")
        except Exception as e:
            print(f"Warning: An error occurred while setting 'NanumGothic'. Details: {e}")

    else:
        print("Warning: Could not determine the OS to set a proper Hangul font. Characters may appear broken.")

    plt.rcParams['axes.unicode_minus'] = False


def save_transaction_plots_by_date(df, output_dir='preprocessed'):
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Starting to extract daily transaction counts ---")
    daily_transaction_counts = df['거래일'].value_counts()
    daily_transaction_counts_sorted = daily_transaction_counts.sort_index()

    plt.figure(figsize=(15, 7))
    daily_transaction_counts_sorted.plot(kind='line', color='royalblue')
    plt.title('거래일 별 거래 건수 추이', fontsize=16)
    plt.xlabel('거래일', fontsize=12)
    plt.ylabel('거래 건수', fontsize=12)
    plt.margins(x=0.01)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    output_viz_path = os.path.join(output_dir, '거래일별_거래건수_추이.png')
    plt.savefig(output_viz_path, dpi=150)
    print(f"Trend chart has been saved to: '{output_viz_path}'")

    plt.close()

def save_top100_plots_by_sido(df, output_dir='preprocessed'):
    os.makedirs(output_dir, exist_ok=True)
    print(f"transcation top 100 chart images are saved in '{output_dir}' folder.")

    sido_list = df['시도명'].unique()

    for sido in sido_list:
        print(f"\n'{sido}' chart is saving...")

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
        plt.ylim(bottom=0)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.margins(x=0.01)
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        safe_sido_name = "".join(c for c in sido if c.isalnum())
        filename = f"{safe_sido_name}_가격추이_상위100.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=150)
        print(f"'{filepath}' file saved.")

        plt.close()

    print("\n--- All Chart is saved. ---")

def combine_images_to_grid(input_dir='preprocessed', output_filename='Combined_Apartment_Trends.png', except_filename='아오지', grid_size=(4, 4)):
    print("\n--- Starting to combine plot images into a single grid image ---")

    try:
        image_files = [
            f for f in os.listdir(input_dir)
            if f.endswith('_가격추이_상위100.png') and not f.startswith(except_filename)
        ]
        image_files.sort()
        print("Excluding files starting with '서울'.")
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
    total_width = img_width * cols
    total_height = img_height * rows

    grid_image = Image.new('RGB', (total_width, total_height), 'white')
    print(f"Creating a new grid image with original size {total_width}x{total_height}...")

    for index, filename in enumerate(image_files):
        if index >= rows * cols:
            print(f"Warning: More than {rows * cols} images found. Only the first {rows * cols} will be included.")
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

    target_width, target_height = 3000, 1600
    print(f"Resizing final image to {target_width}x{target_height}...")
    resized_image = grid_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    output_path = os.path.join(input_dir, output_filename)
    resized_image.save(output_path, dpi=(150, 150))
    print(f"--- Successfully combined and resized {min(len(image_files), rows * cols)} images into '{output_path}' ---")


def plot_macro_trends(pop_df, unemp_df, cpi_df, household_df, bank_df, rate_df, output_dir='preprocessed'):

    # 1. 행정구역별 인구수 추이
    plt.figure(figsize=(15, 8))
    for col in pop_df.columns[1:]:
        plt.plot(pop_df['날짜'], pop_df[col], label=col, alpha=0.7)
    plt.title('행정구역별 인구수 추이', fontsize=16)
    plt.xlabel('날짜')
    plt.ylabel('인구수')
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '행정구역별_인구수_추이.png'), dpi=150)
    plt.close()

    # 2. 행정구역별 실업률 추이
    plt.figure(figsize=(15, 8))
    for col in unemp_df.columns[1:]:
        plt.plot(unemp_df['날짜'], unemp_df[col], label=col, alpha=0.7)
    plt.title('행정구역별 실업률 추이', fontsize=16)
    plt.xlabel('날짜')
    plt.ylabel('실업률(%)')
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '행정구역별_실업률_추이.png'), dpi=150)
    plt.close()

    # 3. 소비자물가지수(CPI) 추이
    plt.figure(figsize=(12, 6))
    plt.plot(cpi_df['날짜'], cpi_df['CPI_총지수'], color='royalblue', label='CPI 총지수')
    plt.title('소비자물가지수(CPI) 추이', fontsize=16)
    plt.xlabel('날짜')
    plt.ylabel('CPI 총지수')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '소비자물가지수_추이.png'), dpi=150)
    plt.close()

    # 4. 지역별 가계대출 추이
    plt.figure(figsize=(15, 8))
    for col in household_df.columns[1:]:
        plt.plot(household_df['날짜'], household_df[col], label=col, alpha=0.7)
    plt.title('지역별 가계대출 추이', fontsize=16)
    plt.xlabel('날짜')
    plt.ylabel('가계대출(만원)')
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '지역별_가계대출_추이.png'), dpi=150)
    plt.close()

    # 5. 지역별 은행대출 추이
    plt.figure(figsize=(15, 8))
    for col in bank_df.columns[1:]:
        plt.plot(bank_df['날짜'], bank_df[col], label=col, alpha=0.7)
    plt.title('지역별 은행대출 추이', fontsize=16)
    plt.xlabel('날짜')
    plt.ylabel('은행대출(만원)')
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '지역별_은행대출_추이.png'), dpi=150)
    plt.close()

    # 6. 한국은행 기준금리 추이
    plt.figure(figsize=(12, 6))
    plt.plot(rate_df['날짜'], rate_df['기준금리'], color='darkred', linewidth=2)
    plt.title('한국은행 기준금리 추이', fontsize=16)
    plt.xlabel('날짜')
    plt.ylabel('기준금리(%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '한국은행_기준금리_추이.png'), dpi=150)
    plt.close()

    print(f"부가데이터 시계열 차트가 '{output_dir}' 폴더에 저장되었습니다.")


if __name__ == "__main__":

    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    set_font()

    apt_price_df = build_apt_price_with_macro()
    print("\nall data merged apartment price data frame:")
    print(apt_price_df)

    output_file = 'KoreaApartDeal_PreProcessed.csv'
    print("\n writing", output_file)
    output_path = os.path.join(PREPROCESSED_DIR, output_file)
    apt_price_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nwritten done: {output_path}")

    output_zip_file = output_file.replace('.csv', '.zip')
    output_zip_path = os.path.join(PREPROCESSED_DIR, output_zip_file)

    print(f"'Compressing {output_file}' to '{output_zip_file}' ...")
    create_split_zip(output_path, output_zip_path, 40)

    print("\n--- Starting Data Visualization ---")

    save_transaction_plots_by_date(apt_price_df, 'preprocessed')
    save_top100_plots_by_sido(apt_price_df, 'preprocessed')

    combine_images_to_grid(input_dir='preprocessed', output_filename='16개지역-top100.png', except_filename='서울', grid_size=(4, 4))

    print("\n--- Finish Data Visualization ---")
