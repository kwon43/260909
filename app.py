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

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

CUTOFF_YEAR = 2025
MIN_DAYS = 300


# ==========================================
# 제목
# ==========================================

st.title("🌡️ 서울 기온 예측기")

st.write(
    "서울의 과거 기온 데이터를 이용하여 연도별 평균기온을 계산하고, "
    "회귀 직선을 이용해 선택한 연도의 예상 평균기온을 보여줍니다."
)


# ==========================================
# 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8-sig"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 필요한 데이터가 없는 행 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 연도 추가
    df["연도"] = df["날짜"].dt.year

    return df


try:

    df = load_data()

except Exception as e:

    st.error("데이터를 불러오는 중 오류가 발생했습니다.")

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
# 관측일 300일 미만인 해 제거
# ==========================================

yearly = yearly[
    yearly["관측일수"] >= MIN_DAYS
].copy()


yearly["연도"] = yearly["연도"].astype(int)

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# ==========================================
# 데이터 확인
# ==========================================

if len(yearly) < 2:

    st.error(
        "회귀 직선을 계산할 수 있는 데이터가 충분하지 않습니다."
    )

    st.stop()


# ==========================================
# 회귀분석
# ==========================================

x = yearly["연도"].to_numpy(
    dtype=float
)

y = yearly["연평균기온"].to_numpy(
    dtype=float
)


# 1차 선형회귀
slope, intercept = np.polyfit(
    x,
    y,
    1
)


# 회귀 직선의 예측값
predicted = (
    slope * x
    + intercept
)


# 상관계수
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# ==========================================
# 회귀에 사용된 기간
# ==========================================

start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

year_count = len(yearly)


# ==========================================
# 연도 슬라이더
# ==========================================

st.subheader("🔎 예상 기온 확인")

selected_year = st.slider(
    "예상할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 선택한 연도의 예상 기온
selected_temp = (
    slope * selected_year
    + intercept
)


# ==========================================
# 예상 기온 크게 표시
# ==========================================

st.markdown(
    f"""
    <div style="
        background-color:#f0f7ff;
        padding:30px;
        border-radius:20px;
        text-align:center;
        margin:20px 0;
    ">

        <div style="
            font-size:25px;
            font-weight:bold;
        ">
            {selected_year}년 예상 연평균기온
        </div>

        <div style="
            font-size:55px;
            font-weight:bold;
            margin-top:10px;
        ">
            {selected_temp:.2f} °C
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# 주요 정보
# ==========================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "회귀에 사용한 연도 수",
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
    f"📊 회귀 직선을 만든 기간은 "
    f"**{start_year}년 ~ {end_year}년**이며, "
    f"총 **{year_count}개 연도**를 사용했습니다."
)


# ==========================================
# 그래프
# ==========================================

st.subheader("📈 연도별 평균기온과 회귀 직선")


fig = go.Figure()


# ------------------------------------------
# 산점도
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
# 회귀 직선
# ------------------------------------------

line_x = np.array(
    [1900, 2100],
    dtype=float
)

line_y = (
    slope * line_x
    + intercept
)


fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,

        mode="lines",

        name="회귀 직선",

        hovertemplate=
            "예상 연평균기온: %{y:.2f} °C"
            "<extra></extra>",

        line=dict(
            width=3
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
# 그래프 디자인
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
# 회귀 분석 결과
# ==========================================

st.subheader("📊 회귀 분석 결과")


st.write(
    f"**회귀식**"
)

st.latex(
    f"y = {slope:.4f}x + {intercept:.2f}"
)


st.write(
    f"**상관계수: {correlation:.3f}**"
)


if correlation > 0:

    st.success(
        "연도가 증가할수록 연평균기온이 높아지는 "
        "경향이 나타납니다. 📈"
    )

elif correlation < 0:

    st.warning(
        "연도가 증가할수록 연평균기온이 낮아지는 "
        "경향이 나타납니다. 📉"
    )

else:

    st.info(
        "연도와 연평균기온 사이에 "
        "뚜렷한 선형 관계가 나타나지 않습니다."
    )


# ==========================================
# 데이터 조건 설명
# ==========================================

st.subheader("📌 데이터 처리 기준")

st.write(
    f"""
    - 기준 기간: **{CUTOFF_YEAR}년까지**
    - {CUTOFF_YEAR}년 이후 자료: **제외**
    - 관측일수가 **{MIN_DAYS}일 미만인 연도: 제외**
    - 연평균기온: 해당 연도의 일평균기온을 이용하여 계산
    - 회귀 직선: 남은 연도들의 연평균기온을 이용한 1차 선형회귀
    """
)


# ==========================================
# 사용 데이터 표
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
    "※ 예상 기온은 과거 연도와 연평균기온 사이의 "
    "선형 관계를 이용한 단순 회귀 예측값입니다. "
    "실제 미래 기온을 보장하는 값은 아닙니다."
)
