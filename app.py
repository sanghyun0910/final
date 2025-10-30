# -*- coding: utf-8 -*-
import os
import time
import re
import threading
import math
from collections import deque
from datetime import datetime

import streamlit as st
import pandas as pd
import serial
import serial.tools.list_ports
from streamlit_autorefresh import st_autorefresh

# ---------------------------
# 앱 기본 설정
# ---------------------------
st.set_page_config(page_title="DHT22 대시보드 + 제습 가이드", layout="wide")

# ---------------------------
# 유틸 함수
# ---------------------------
def list_serial_ports():
    """사용 가능한 시리얼 포트 목록"""
    try:
        ports = serial.tools.list_ports.comports()
        return [p.device for p in ports]
    except Exception:
        return []

def parse_line(line: str):
    """
    장치 출력 파싱 → (온도°C, 습도%) 튜플 반환
    - 기본 가정: 'OK,습도,온도'
    - 백업: 라인에서 숫자 2개 추출 후 (습도,온도)로 가정하여 교정
    """
    try:
        s = line.strip()
        if "," in s:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 3 and parts[0].upper() == "OK":
                h = float(parts[1])
                t = float(parts[2])
                return t, h
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        if len(nums) >= 2:
            h = float(nums[0])
            t = float(nums[1])
            return t, h
    except Exception:
        pass
    return None

def absolute_humidity(T_c: float, RH_pct: float) -> float:
    """
    절대습도 AH (g/m³)
    T_c: 섭씨온도(°C), RH_pct: 상대습도(%)
    """
    return 216.7 * ((RH_pct / 100.0) * 6.112 * math.exp((17.62 * T_c) / (243.12 + T_c))) / (273.15 + T_c)

# ---------------------------
# 전역 리소스: 시리얼 & 수집 스레드 (싱글톤)
# ---------------------------
@st.cache_resource
def get_serial_worker():
    """
    프로세스 전역 1개만 유지되는 상태/스레드
    - ser: 시리얼 핸들
    - port, baud: 설정
    - running: 루프 플래그
    - buf: 최근 데이터 (deque of (ts, t, h))
    - lock: 버퍼 보호
    - log_path: CSV 경로
    - thread: 백그라운드 리더
    """
    state = {
        "ser": None,
        "port": None,
        "baud": 9600,
        "running": True,
        "buf": deque(maxlen=10000),
        "lock": threading.Lock(),
        "log_path": "dht22_log.csv",
        "thread": None,
    }

    def ensure_open():
        """포트 열기 (설정이 있을 때만)"""
        if state["ser"] and state["ser"].is_open:
            return
        if not state["port"]:
            return
        try:
            s = serial.Serial(state["port"], baudrate=state["baud"], timeout=2)
            s.reset_input_buffer()
            state["ser"] = s
        except Exception:
            state["ser"] = None

    def reader_loop():
        """시리얼에서 한 줄씩 읽어 전역 버퍼/CSV에 누적"""
        while state["running"]:
            try:
                ensure_open()
                if not (state["ser"] and state["ser"].is_open):
                    time.sleep(1.0)
                    continue

                line = state["ser"].readline().decode("utf-8", errors="ignore")
                parsed = parse_line(line)
                if parsed:
                    t, h = parsed
                    ts = datetime.now()
                    with state["lock"]:
                        state["buf"].append((ts, t, h))
                    # CSV append
                    try:
                        header_needed = not os.path.exists(state["log_path"])
                        pd.DataFrame([[ts, t, h]], columns=["time", "temp_c", "humid"]).to_csv(
                            state["log_path"], mode="a", index=False, header=header_needed
                        )
                    except Exception:
                        pass
                else:
                    time.sleep(0.05)
            except Exception:
                time.sleep(0.5)

    th = threading.Thread(target=reader_loop, daemon=True)
    state["thread"] = th
    th.start()
    return state

# ---------------------------
# 사이드바: 연결/환경 설정
# ---------------------------
state = get_serial_worker()
st.sidebar.title("연결 설정 (서버 전역)")
available_ports = list_serial_ports()
default_idx = 0 if available_ports else None
port = st.sidebar.selectbox("시리얼 포트", available_ports, index=default_idx)
baud = st.sidebar.number_input("Baudrate", value=state["baud"], step=1200)
interval_sec = st.sidebar.number_input("새로고침(초)", value=2, min_value=1, step=1)

st.sidebar.markdown("---")
log_toggle = st.sidebar.checkbox("CSV 로깅", value=True)
if log_toggle:
    log_path = st.sidebar.text_input("CSV 경로", value=state["log_path"])
else:
    log_path = None

col_a, col_b = st.sidebar.columns(2)
connect_btn = col_a.button("연결/재연결")
disconnect_btn = col_b.button("해제")

if connect_btn:
    state["port"] = port
    state["baud"] = int(baud)
    if log_path:
        state["log_path"] = log_path
    try:
        if state["ser"] and state["ser"].is_open:
            state["ser"].close()
        state["ser"] = None
    except Exception:
        state["ser"] = None
    st.success(f"전역 연결 시도: {state['port']} @ {state['baud']}")

if disconnect_btn:
    try:
        if state["ser"] and state["ser"].is_open:
            state["ser"].close()
        state["ser"] = None
        st.warning("전역 연결 해제됨")
    except Exception as e:
        st.error(f"해제 중 오류: {e}")

# ---------------------------
# 헤더 & 자동 새로고침
# ---------------------------
st.title("🌡️ DHT22 실시간 대시보드 + 제습 가이드")
st.caption("Streamlit + PySerial + DHT22 + Dehumidifier Guide (멀티클라이언트 공유)")

st_autorefresh(interval=int(interval_sec * 1000), key="autorefresh")

# ---------------------------
# 전역 버퍼 → DataFrame
# ---------------------------
with state["lock"]:
    data = list(state["buf"])
df = pd.DataFrame(data, columns=["time", "temp_c", "humid"]) if data else pd.DataFrame(columns=["time", "temp_c", "humid"])

# ---------------------------
# 현재 상태 표시
# ---------------------------
ser_open = bool(state["ser"] and state["ser"].is_open)
st.info(f"연결 상태: {'✅ 연결됨' if ser_open else '⚠️ 미연결'} | 포트: {state.get('port')} | Baud: {state.get('baud')}")

col1, col2 = st.columns(2)
if len(df):
    curr_t = float(df["temp_c"].iloc[-1])
    curr_h = float(df["humid"].iloc[-1])
    delta_t = (curr_t - df["temp_c"].iloc[-2]) if len(df) >= 2 else None
    delta_h = (curr_h - df["humid"].iloc[-2]) if len(df) >= 2 else None
else:
    curr_t = curr_h = delta_t = delta_h = None

col1.metric("현재 온도 (°C)", f"{curr_t:.2f}" if curr_t is not None else "-", f"{delta_t:+.2f}" if delta_t is not None else None)
col2.metric("현재 습도 (%)", f"{curr_h:.2f}" if curr_h is not None else "-", f"{delta_h:+.2f}" if delta_h is not None else None)

st.markdown("---")

# ---------------------------
# 차트 & 테이블
# ---------------------------
if len(df):
    df_plot = df.set_index("time")
    st.line_chart(df_plot[["temp_c", "humid"]])
    with st.expander("📋 최근 데이터 (마지막 30개)"):
        st.dataframe(df.tail(30), use_container_width=True)
else:
    st.info("데이터가 없습니다. 장치가 데이터를 전송 중인지 확인하세요.")

# ---------------------------
# 💧 제습기 작동 가이드 (기본)
# ---------------------------
st.markdown("## 💧 제습기 작동 가이드")
room_size = st.number_input("실내 면적 (m²)", value=20.0, min_value=5.0)
ceiling_height = st.number_input("천장 높이 (m)", value=2.4)
target_humid = st.slider("목표 습도 (%)", 40, 60, 50)

if curr_t is not None and curr_h is not None:
    AH_current = absolute_humidity(curr_t, curr_h)
    AH_target = absolute_humidity(curr_t, target_humid)
    volume_m3 = room_size * ceiling_height
    delta_W_g = (AH_current - AH_target) * volume_m3
    delta_L = max(0.0, delta_W_g / 1000.0)

    if delta_L <= 0:
        st.success("✅ 이미 목표 습도 이하입니다. 제습기 필요 없음.")
    else:
        st.caption(f"제거해야 할 수분량: {delta_L:.2f} L (절대습도 차 {AH_current - AH_target:.2f} g/m³)")
else:
    st.info("온도·습도 데이터가 필요합니다.")

# ---------------------------
# ⚙️ 기기 스펙/요금/탄소 계산
# ---------------------------
st.markdown("## ⚙️ 기기 스펙 & 비용/탄소 계산")

with st.expander("기기 스펙 (라벨값 기본)"):
    # 사진 기준: 정격 제습 17 L/일, 소비전력 390 W, 효율 1.86 L/kWh, CO₂ 165 g/시간
    rated_capacity_L_per_day = st.number_input("정격 제습량 (L/일)", value=17.0, min_value=1.0)
    rated_power_kW = st.number_input("소비전력 (kW)", value=0.39, step=0.01, format="%.2f")
    efficiency_L_per_kWh = st.number_input("제습효율 (L/kWh)", value=1.86, step=0.01)
    co2_g_per_hour = st.number_input("CO₂ 배출 (g/시간, 라벨값)", value=165.0, step=1.0)

with st.expander("요금/환경 설정"):
    elec_cost_per_kWh = st.number_input("전기요금 단가 (원/kWh)", value=140.0, step=10.0)

# 목표 도달 시간/비용/탄소 계산 (기본 요약)
if curr_t is not None and curr_h is not None:
    AH_now = absolute_humidity(curr_t, curr_h)
    AH_target = absolute_humidity(curr_t, target_humid)
    volume_m3 = room_size * ceiling_height
    delta_L = max(0.0, (AH_now - AH_target) * volume_m3 / 1000.0)

    if delta_L > 0:
        L_per_hour = rated_capacity_L_per_day / 24.0
        hours_needed = delta_L / L_per_hour

        kWh = hours_needed * rated_power_kW
        cost_won = kWh * elec_cost_per_kWh
        co2_g = hours_needed * co2_g_per_hour

        colA, colB, colC, colD = st.columns(4)
        colA.metric("필요 제거수분량", f"{delta_L:.2f} L")
        colB.metric("예상 작동시간", f"{hours_needed:.1f} h")
        colC.metric("전기요금", f"{cost_won:,.0f} 원")
        colD.metric("탄소배출", f"{co2_g:,.0f} gCO₂")

        st.caption(
            f"효율 참고: 1 kWh당 약 {efficiency_L_per_kWh:.2f} L 제거 (라벨). "
            f"현재 추정 제거속도 {L_per_hour:.2f} L/h"
        )
    else:
        st.info("현재 상태는 추가 제습이 필요하지 않습니다.")

# =========================================================
# ⏱ 목표 습도 도달 '운영시간' 전용 섹션 (추가)
# =========================================================
st.markdown("## ⏱ 목표 습도 도달 예상 운영시간")

def estimate_runtime_to_target(
    temp_c: float,
    rh_now: float,
    rh_target: float,
    room_m2: float,
    height_m: float,
    rated_L_per_day: float,
    power_kW: float,
    price_per_kWh: float,
    co2_gph: float,
    safety_margin: float = 1.0,
):
    """
    temp_c: 현재 온도(°C)
    rh_now: 현재 상대습도(%)
    rh_target: 목표 상대습도(%)
    room_m2: 실내 면적(m²)
    height_m: 천장 높이(m)
    rated_L_per_day: 제습기 정격 제습량(L/day)
    power_kW: 소비전력(kW)
    price_per_kWh: 전기요금 단가(원/kWh)
    co2_gph: 시간당 CO2 배출량(g/h) - 선택
    safety_margin: 여유율(누수/문 열림/흡습체 등 변수 대비, 1.0 = 0% 여유)
    """
    # 절대습도 계산
    def AH(T_c, RH_pct):
        return 216.7 * ((RH_pct / 100.0) * 6.112 * math.exp((17.62 * T_c) / (243.12 + T_c))) / (273.15 + T_c)

    AH_now = AH(temp_c, rh_now)
    AH_tgt = AH(temp_c, rh_target)

    volume_m3 = room_m2 * height_m
    delta_L = max(0.0, (AH_now - AH_tgt) * volume_m3 / 1000.0)  # g→kg→L (1kg ≈ 1L)
    delta_L *= safety_margin  # 여유율 적용

    L_per_hour = max(1e-9, rated_L_per_day / 24.0)  # 0 나눗셈 방지
    hours_needed = delta_L / L_per_hour

    kWh = hours_needed * power_kW
    cost_won = kWh * price_per_kWh
    co2_g = hours_needed * co2_gph if co2_gph is not None else None

    return {
        "AH_now": AH_now,
        "AH_tgt": AH_tgt,
        "delta_L": delta_L,
        "L_per_hour": L_per_hour,
        "hours": hours_needed,
        "kWh": kWh,
        "cost_won": cost_won,
        "co2_g": co2_g,
    }

# 입력(슬라이더/숫자) - 기존 변수 재사용
safety_margin = st.slider("여유율(누수/환기 변수 대비)", 1.0, 1.5, 1.2, step=0.05)

if (curr_t is not None) and (curr_h is not None):
    result = estimate_runtime_to_target(
        temp_c=curr_t,
        rh_now=curr_h,
        rh_target=target_humid,
        room_m2=room_size,
        height_m=ceiling_height,
        rated_L_per_day=rated_capacity_L_per_day,
        power_kW=rated_power_kW,
        price_per_kWh=elec_cost_per_kWh,
        co2_gph=co2_g_per_hour if "co2_g_per_hour" in locals() else None,
        safety_margin=safety_margin,
    )

    # 카드 형태 요약
    ca, cb, cc, cd = st.columns(4)
    ca.metric("필요 제거 수분량", f"{result['delta_L']:.2f} L")
    cb.metric("정격 제습속도", f"{result['L_per_hour']:.2f} L/h")
    cc.metric("예상 운영시간", f"{result['hours']:.1f} h")
    cd.metric("예상 전력소비", f"{result['kWh']:.2f} kWh")
    st.caption(f"예상 전기요금: {result['cost_won']:,.0f} 원" + (f" · 추정 탄소배출: {result['co2_g']:,.0f} gCO₂" if result['co2_g'] is not None else ""))

    # 완료 예상 시각
    eta = datetime.now() + pd.to_timedelta(result["hours"], unit="h")
    st.info(f"목표 {target_humid}% 도달 예상 시각: **{eta.strftime('%Y-%m-%d %H:%M')}**")

    # 상세 설명(절대습도 차 등)
    st.caption(
        f"절대습도 차: {(result['AH_now'] - result['AH_tgt']):.2f} g/m³ · "
        f"실내부피: {room_size * ceiling_height:.1f} m³ · "
        f"계산 여유율: ×{safety_margin:.2f}"
    )
else:
    st.warning("현재 온도/습도 데이터가 필요합니다. (장치 연결 및 수신 확인)")

# ---------------------------
# 🧭 추천 운전 모드 (ΔH/Δt 기반)
# ---------------------------
st.markdown("## 🧭 추천 운전 모드")

if curr_h is not None and len(df) >= 5:
    # 최근 N분 변화속도(%/분) 계산
    lookback_min = st.slider("변화속도 분석 구간(분)", 5, 30, 15)
    t_cut = pd.Timestamp.now() - pd.Timedelta(minutes=lookback_min)
    recent = df[df["time"] >= t_cut]
    if len(recent) >= 5:
        dh = float(recent["humid"].iloc[-1] - recent["humid"].iloc[0])
        dt_min = (recent["time"].iloc[-1] - recent["time"].iloc[0]).total_seconds() / 60.0
        slope = dh / dt_min if dt_min > 0 else 0.0  # %/분
    else:
        slope = 0.0

    # 규칙 기반 추천
    if curr_h <= 55:
        mode = "OFF"
        advice = "이상적 구간입니다. 필요 시 짧은 환기만 유지하세요."
    elif curr_h <= 65 and slope <= 0.1:
        mode = "Quiet"
        advice = "약 1~2시간 저속 운전 권장."
    elif curr_h <= 75 or slope > 0.1:
        mode = "Eco"
        advice = "2~4시간 중속 운전 권장. 창문 틈새 등 누기 점검."
    else:
        mode = "Boost"
        advice = "연속 운전 + 5~10분 환기 추천. 빨래건조 모드 고려."

    colm1, colm2 = st.columns(2)
    colm1.info(f"현재 습도 {curr_h:.1f}% · 최근 변화속도 {slope:+.2f}%/분")
    colm2.success(f"권장 모드: **{mode}** — {advice}")
else:
    st.info("변화속도 분석에는 최근 데이터가 조금 더 필요합니다.")

# ---------------------------
# 🧲 (선택) 자동제어 히스테리시스 (릴레이 연동 지점)
# ---------------------------
with st.expander("자동제어 (선택, 릴레이/스마트플러그 연동용)"):
    auto_enable = st.checkbox("목표 기반 자동 ON/OFF", value=False)
    on_th = st.slider("ON 임계(%)", 45, 70, 52)
    off_th = st.slider("OFF 임계(%)", 35, 65, 48)
    st.caption("예: 52% 이상이면 ON, 48% 이하 떨어지면 OFF → 잦은 스위칭 방지")

    if auto_enable and curr_h is not None:
        # 실제 제어 코드 위치:
        # if curr_h >= on_th: relay_on()
        # elif curr_h <= off_th: relay_off()
        if curr_h >= on_th:
            relay_state = "ON"
        elif curr_h <= off_th:
            relay_state = "OFF"
        else:
            relay_state = "HOLD"
        st.warning(f"릴레이 상태(가상): {relay_state}")
        st.caption("주의: 220V 기기 제어 시 절연/정격 준수 및 안전 장비 필수!")

# ---------------------------
# 푸터
# ---------------------------
st.markdown("---")
st.caption(
    "Tips: 실내 부피(면적×층고), 흡습체(이불/옷장) 많으면 시간에 20~30% 여유를 두세요. "
    "야간전력/시간대 요금제를 쓰면 단가 입력값을 조정해 실제 비용을 반영할 수 있습니다."
)
