import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
import os

# --- 1. 頁面設定 ---
st.set_page_config(page_title="智慧嬰語翻譯機 (研究版 v2.0)", page_icon="🔬", layout="wide")

# 自訂 CSS
st.markdown("""
    <style>
    .stButton>button { width: 100%; height: 70px; font-size: 20px; font-weight: bold; border-radius: 15px; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #6c757d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🔬 智慧嬰語翻譯機 (研究優化版)")
st.caption("基於 Zeskind (1997) 與 Dunstan 聲學特徵模型")

# --- 2. 側邊欄：情境變數 ---
with st.sidebar:
    st.header("📊 參數校正")
    st.write("針對 4 個月大嬰兒優化")
    last_feed = st.slider("距離上一餐 (小時)", 0.0, 6.0, 2.5, 0.5)
    diaper_status = st.radio("尿布狀態", ["乾淨", "髒/濕"])
    
    st.markdown("---")
    st.info("**科學指標說明：**\n\n1. **F0 (基頻)**: 疼痛哭聲通常 > 450Hz\n2. **規律性**: 飢餓哭聲能量起伏大\n3. **ZCR (過零率)**: 聲音越沙啞/尖銳數值越高")

# --- 3. 錄音區 ---
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("採樣控制")
    st.write("請錄製約 3-5 秒")
with col2:
    audio = mic_recorder(
        start_prompt="🔴 開始錄音 (Start)",
        stop_prompt="⬛ 停止並分析 (Stop)",
        key='recorder',
        format='wav'
    )

# --- 4. 核心分析邏輯 ---
if audio:
    audio_bytes = audio['bytes']
    temp_filename = "cloud_upload.wav"
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)

    st.success("✅ 音訊接收成功，正在進行頻譜分析...")
    
    try:
        # A. 載入音訊
        y, sr = librosa.load(temp_filename)
        
        # B. 提取高階聲學特徵
        
        # 1. 能量特徵 (RMS)
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)
        rms_std = np.std(rms)  # 能量標準差 (判斷規律性關鍵)
        
        # 2. 頻率特徵 (Spectral Centroid & ZCR)
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = np.mean(centroids)
        
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        avg_zcr = np.mean(zcr) # 過零率 (判斷噪音/沙啞度)
        
        # 3. 節奏特徵 (Onset & Tempo)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        bpm = tempo[0]

        # C. 視覺化儀表板
        with st.expander("📈 點擊查看詳細聲學波形", expanded=True):
            fig, ax = plt.subplots(figsize=(12, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax, color='#17a2b8')
            ax.set_title(f"Oscillogram (Energy Variance: {rms_std:.4f})")
            st.pyplot(fig)

        # D. 數據矩陣
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("能量均值 (Intensity)", f"{avg_rms:.3f}")
        c2.metric("頻譜質心 (Pitch)", f"{avg_centroid:.0f} Hz")
        c3.metric("規律性 (Rhythm)", f"{rms_std:.3f}", delta_color="inverse")
        c4.metric("尖銳度 (ZCR)", f"{avg_zcr:.3f}")

        # E. 決策樹邏輯 (基於研究文獻優化)
        # 這些閾值是基於一般 3-6 個月嬰兒數據設定
        
        prediction = "未知"
        reason = ""
        color = "gray"

        # 邏輯 1: 疼痛 (Pain) - 高能量、高頻、持續無規律
        if avg_rms > 0.1 and avg_centroid > 3000:
            prediction = "疼痛 (Pain)"
            reason = "偵測到極高頻尖叫與高能量，且聲音緊繃 (High Centroid)。"
            color = "red"
            
        # 邏輯 2: 飢餓 (Hunger) - 高規律性 (Rhythmic)、中高能量
        # rms_std > 0.03 代表聲音忽大忽小（有換氣節奏）
        elif rms_std > 0.03 and avg_rms > 0.05:
            prediction = "飢餓 (Hunger)"
            reason = "能量波形呈現高度規律性 (High Variance)，符合飢餓哭聲特徵。"
            color = "orange"
            
        # 邏輯 3: 疲倦 (Tired) - 低能量、音調遞減
        elif avg_rms < 0.05:
            prediction = "疲倦 (Tired)"
            reason = "整體能量較低，聲音拖長且無爆發力。"
            color = "blue"
            
        # 邏輯 4: 不適/撒嬌 (Discomfort) - 高ZCR但能量中等
        elif avg_zcr > 0.1:
            prediction = "不適/脹氣 (Discomfort)"
            reason = "聲音聽起來較為煩躁沙啞 (High ZCR)，可能是尿布濕或脹氣。"
            color = "green"
            
        else:
            prediction = "尋求關注 (Attention)"
            reason = "各項數值均衡，可能是無聊或想要抱抱。"
            color = "green"

        # F. 綜合診斷報告
        st.divider()
        st.subheader(f"AI 診斷結果: :{color}[{prediction}]")
        st.write(f"**聲學判讀依據:** {reason}")
        
        # G. SOP 建議
        st.markdown("### 🛡️ 安全主任建議行動 (SOP)")
        
        if "疼痛" in prediction:
            st.error("""
            **緊急應變程序:**
            1. **檢查身體:** 確認無外傷、頭髮纏繞 (Hair Tourniquet)。
            2. **排除病理:** 觀察是否有發燒、嘔吐或腹股溝疝氣徵兆。
            3. **腸絞痛檢測:** 若發生於黃昏且持續尖叫，觸診腹部是否僵硬。
            """)
        elif "飢餓" in prediction:
            if last_feed < 2:
                 st.warning(f"雖然聲學特徵像飢餓，但距離上一餐僅 {last_feed} 小時。建議先檢查**脹氣**或給予**安撫奶嘴** (滿足口慾)。")
            else:
                 st.success("**建議行動:** 立即準備餵食。")
        elif "疲倦" in prediction:
             st.info("**建議行動:** 執行睡眠儀式 (關燈、白噪音、包巾)，避免過度逗弄。")
        else:
             st.success("**建議行動:** 檢查尿布，或變換抱姿 (足球抱/飛機抱) 緩解不適。")

        # 清理暫存
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    except Exception as e:
        st.error(f"分析失敗: {e}")
else:
    st.info("等待輸入...")
