import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ==========================================
# 기본 설정
# ==========================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

CUTOFF_YEAR = 2025
MIN_DAYS = 300


# ==========================================
# 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    df["연도"] = df["날짜"].dt.year

    return df


# ==========================================
# 데이터 불러오기 오류 처리
# ==========================================

try:

    df = load_data()

except Exception as e:

    st.error("❌ 데이터를 불러오는 중 오류가 발생했습니다.")

    st.exception(e)

    st.stop()


# ==========================================
# 연도별 평균기온 계산
# ==========================================

yearly = (
    df[df["연도"] <= CUTOFF_YEAR]
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# ==========================================
# 관측일 300일 미만인 연도 제외
# ==========================================

yearly = yearly[
    yearly["관측일수"] >= MIN_DAYS
].copy()

yearly["연도"] = yearly["연도"].astype(int)

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# ==========================================
# 데이터가 부족한 경우
# ==========================================

if len(yearly) < 2:

    st.error(
        "회귀 분석을 할 수 있는 데이터가 충분하지 않습니다."
    )

    st.stop()


# ==========================================
# 전체 기간 회귀분석
# ==========================================

x_all = yearly["연도"].to_numpy(
    dtype=float
)

y_all = yearly["연평균기온"].to_numpy(
    dtype=float
)


slope_all, intercept_all = np.polyfit(
    x_all,
    y_all,
    1
)


# 상관계수
correlation = np.corrcoef(
    x_all,
    y_all
)[0, 1]


# ==========================================
# 전체 기간 정보
# ==========================================

start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

year_count = len(yearly)


# ==========================================
# 최근 20년 데이터
# ==========================================

recent_start_year = end_year - 19

recent = yearly[
    yearly["연도"] >= recent_start_year
].copy()


if len(recent) >= 2:

    x_recent = recent["연도"].to_numpy(
        dtype=float
    )

    y_recent = recent["연평균기온"].to_numpy(
        dtype=float
    )

    slope_recent, intercept_recent = np.polyfit(
        x_recent,
        y_recent,
        1
    )

else:

    slope_recent = np.nan
    intercept_recent = np.nan


# ==========================================
# 100년에 몇 도 오르는가 계산
# ==========================================

rise_all_100 = slope_all * 100


if not np.isnan(slope_recent):

    rise_recent_100 = slope_recent * 100

else:

    rise_recent_100 = np.nan


# ==========================================
# 제목
# ==========================================

st.title("🌡️ 기온 예측기")

st.write(
    "서울의 연평균기온 데이터를 이용하여 "
    "기온 변화 추세를 분석하고 미래의 예상 기온을 확인합니다."
)


# ==========================================
# 기온 상승 속도
# ==========================================

st.header("🔥 기온 상승 속도")

st.metric(
    label="전체 기간 기준 100년에 몇 °C 오르는가",
    value=f"{rise_all_100:+.2f} °C"
)


# ==========================================
# 전체 기간 vs 최근 20년
# ==========================================

st.header("📊 전체 기간과 최근 20년 비교")

col1, col2 = st.columns(2)


with col1:

    st.subheader("🌏 전체 기간")

    st.metric(
        label="100년 기준 기온 변화",
        value=f"{rise_all_100:+.2f} °C"
    )

    st.write(
        f"사용 기간: **{start_year}년 ~ {end_year}년**"
    )

    st.write(
        f"사용 연도: **{year_count}개**"
    )


with col2:

    st.subheader("🔥 최근 20년")

    if not np.isnan(rise_recent_100):

        st.metric(
            label="100년 기준 기온 변화",
            value=f"{rise_recent_100:+.2f} °C"
        )

        st.write(
            f"사용 기간: **{recent_start_year}년 ~ {end_year}년**"
        )

        st.write(
            f"사용 연도: **{len(recent)}개**"
        )

    else:

        st.warning(
            "최근 20년 데이터가 충분하지 않습니다."
        )


# ==========================================
# 상승 속도 차이
# ==========================================

if not np.isnan(rise_recent_100):

    difference = (
        rise_recent_100
        - rise_all_100
    )

    st.subheader("🔍 상승 속도 차이")

    if difference > 0:

        st.success(
            f"최근 20년의 상승 속도가 전체 기간보다 "
            f"100년 기준 **{difference:.2f}°C 더 빠릅니다.**"
        )

    elif difference < 0:

        st.info(
            f"최근 20년의 상승 속도가 전체 기간보다 "
            f"100년 기준 **{abs(difference):.2f}°C 더 느립니다.**"
        )

    else:

        st.info(
            "최근 20년과 전체 기간의 상승 속도가 같습니다."
        )


# ==========================================
# 예상 연도 선택
# ==========================================

st.header("🔎 기온 예측")

selected_year = st.slider(
    "예상할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 전체 기간 회귀식으로 예상
selected_temp = (
    slope_all * selected_year
    + intercept_all
)


# ==========================================
# 예상 기온
# ==========================================

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{selected_temp:.2f} °C"
)


# ==========================================
# 주요 정보
# ==========================================

st.header("📌 회귀 분석에 사용된 데이터")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "사용 연도 수",
        f"{year_count}개"
    )


with col2:

    st.metric(
        "시작 연도",
        f"{start_year}년"
    )


with col3:

    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


with col4:

    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )


st.info(
    f"📈 회귀 직선을 만든 기간은 "
    f"**{start_year}년 ~ {end_year}년**이며 "
    f"총 **{year_count}개 연도**를 사용했습니다."
)


# ==========================================
# 산점도 + 회귀 직선
# ==========================================

st.header("📈 연도별 평균기온과 회귀 직선")

fig = go.Figure()


# ------------------------------------------
# 실제 연평균기온 산점도
# ------------------------------------------

fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="실제 연평균기온",

        customdata=yearly["관측일수"],

        hovertemplate=
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata}일"
            "<extra></extra>",

        marker=dict(
            size=7
        )
    )
)


# ------------------------------------------
# 전체 기간 회귀 직선
# ------------------------------------------

line_x = np.array(
    [1900, 2100],
    dtype=float
)

line_y_all = (
    slope_all * line_x
    + intercept_all
)


fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y_all,
        mode="lines",
        name="전체 기간 회귀 직선",

        hovertemplate=
            "예상 연평균기온: %{y:.2f} °C"
            "<extra></extra>",

        line=dict(
            width=3
        )
    )
)


# ------------------------------------------
# 최근 20년 회귀 직선
# ------------------------------------------

if not np.isnan(slope_recent):

    recent_line_x = np.array(
        [recent_start_year, end_year],
        dtype=float
    )

    recent_line_y = (
        slope_recent * recent_line_x
        + intercept_recent
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_x,
            y=recent_line_y,
            mode="lines",
            name="최근 20년 회귀 직선",

            hovertemplate=
                "최근 20년 예상값: %{y:.2f} °C"
                "<extra></extra>",

            line=dict(
                width=3,
                dash="dash"
            )
        )
    )


# ------------------------------------------
# 선택한 연도의 예상값
# ------------------------------------------

fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[selected_temp],
        mode="markers",

        name=f"{selected_year}년 예상값",

        marker=dict(
            size=16,
            symbol="star"
        ),

        hovertemplate=
            f"<b>{selected_year}년</b><br>"
            f"예상 연평균기온: "
            f"{selected_temp:.2f} °C"
            "<extra></extra>"
    )
)


# ==========================================
# 그래프 설정
# ==========================================

fig.update_layout(
    title="서울 연도별 평균기온과 회귀 직선",

    xaxis_title="연도",

    yaxis_title="연평균기온 (°C)",

    hovermode="closest",

    template="plotly_white",

    height=600,

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# 회귀식
# ==========================================

st.header("📐 회귀 분석 결과")

st.write("전체 기간 회귀식")

st.latex(
    f"y = {slope_all:.4f}x + {intercept_all:.2f}"
)


if not np.isnan(slope_recent):

    st.write("최근 20년 회귀식")

    st.latex(
        f"y = {slope_recent:.4f}x + {intercept_recent:.2f}"
    )


st.write(
    f"상관계수: **{correlation:.3f}**"
)


# ==========================================
# 데이터 처리 기준
# ==========================================

st.header("📋 데이터 처리 기준")

st.write(
    f"""
    - 기준 기간: **2025년까지**
    - 2025년 이후 자료: **제외**
    - 관측일수가 **300일 미만인 연도: 제외**
    - 연평균기온: 해당 연도의 일평균기온 평균
    - 전체 기간 회귀: 조건을 만족한 모든 연도 사용
    - 최근 20년 회귀: **{recent_start_year}년 ~ {end_year}년**
    """
)


# ==========================================
# 연도별 데이터
# ==========================================

with st.expander("📋 연도별 데이터 보기"):

    display_df = yearly.copy()

    display_df["연평균기온"] = (
        display_df["연평균기온"]
        .round(2)
    )

    display_df = display_df.rename(
        columns={
            "연도": "연도",
            "연평균기온": "연평균기온 (°C)",
            "관측일수": "관측일수 (일)"
        }
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# 주의사항
# ==========================================

st.caption(
    "※ 예상 기온은 과거 연평균기온과 연도의 선형 관계를 이용한 "
    "단순 회귀 예측값입니다. 실제 미래 기온을 보장하는 값은 아닙니다."
)
