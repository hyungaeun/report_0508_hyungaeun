import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Failure 분석", page_icon="🫀", layout="wide")

st.title("🫀 Heart Failure 데이터 분석")

# ── 데이터 로드 ──────────────────────────────────────────
@st.cache_data
def load_data():
    df_a = pd.read_json('heart_failure_a.json')
    df_b = pd.read_json('heart_failure_b.json')
    df = pd.merge(df_a, df_b, on='person_id', how='inner')
    dropped_num = len(df_a) - len(df) + len(df_b) - len(df)
    return df_a, df_b, df, dropped_num

df_a, df_b, df, dropped_num = load_data()

st.info(f"✅ 데이터 로드 완료! df_a: {len(df_a)}행, df_b: {len(df_b)}행, 병합 후: {len(df)}행 | 사라진 데이터: **{dropped_num}개**")

st.divider()

# ── 그래프 1 : Jointplot ─────────────────────────────────
st.subheader("📊 1. 박출계수(ejection_fraction)와 나이(age)의 상관관계")
st.caption("seaborn jointplot — hue: DEATH_EVENT")

fig1 = sns.jointplot(df, x='ejection_fraction', y='age', hue='DEATH_EVENT')
st.pyplot(fig1.figure)
plt.close()

st.divider()

# ── 그래프 2 : Violinplot (라디오 버튼) ──────────────────
st.subheader("🎻 2. 혈소판(platelets)과 사망(DEATH_EVENT)의 관계")
st.caption("흡연 여부를 라디오 버튼으로 선택하세요")

smoking_option = st.radio(
    "흡연 여부 선택",
    options=["전체 (split=True)", "비흡연자만 (smoking=0)", "흡연자만 (smoking=1)"],
    horizontal=True
)

fig2, ax2 = plt.subplots(figsize=(8, 5))

if smoking_option == "전체 (split=True)":
    sns.violinplot(data=df, x='DEATH_EVENT', y='platelets', hue='smoking', split=True, ax=ax2)
elif smoking_option == "비흡연자만 (smoking=0)":
    sns.violinplot(data=df[df['smoking'] == 0], x='DEATH_EVENT', y='platelets', ax=ax2)
    ax2.set_title("비흡연자 (smoking=0)")
else:
    sns.violinplot(data=df[df['smoking'] == 1], x='DEATH_EVENT', y='platelets', ax=ax2)
    ax2.set_title("흡연자 (smoking=1)")

st.pyplot(fig2)
plt.close()

st.divider()

# ── 그래프 3 : Histplot (슬라이더) ──────────────────────
st.subheader("📈 3. Time 컬럼 히스토그램")
st.caption("ejection_fraction 범위를 슬라이더로 선택하세요")

ef_min = int(df['ejection_fraction'].min())
ef_max = int(df['ejection_fraction'].max())

ef_range = st.slider(
    "ejection_fraction 범위",
    min_value=ef_min,
    max_value=ef_max,
    value=(ef_min, ef_max)
)

df_filtered = df[(df['ejection_fraction'] >= ef_range[0]) & (df['ejection_fraction'] <= ef_range[1])]
st.caption(f"선택된 범위: {ef_range[0]} ~ {ef_range[1]} | 해당 데이터 수: {len(df_filtered)}행")

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.histplot(data=df_filtered, x='time', bins=20, hue='DEATH_EVENT', ax=ax3)
st.pyplot(fig3)
plt.close()
