import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder
import os

# --- 1. 頁面設定 (Page Config) ---
st.set_page_config(page_title="智慧嬰語翻譯機 (雲端版)", page_icon="👶")

# --- 2. 自訂 CSS (優化手機操作體驗) ---
# 加大按鈕尺寸，方便手機點擊
st.markdown("""
    <style>
    .stButton>button { 
        width: 100%; 
        height: 60px; 
        font-size: 18px; 
        font-weight: bold; 
        border-radius: 12px;
    }
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 主標題區 ---
st.title("👶 智慧嬰語翻譯機")
st.caption("SIICAI - Cloud Analysis System")
st.info("說明：請點擊下方紅色按鈕開始錄音，再次點擊即可停止並開始分析。")

# --- 4. 側邊欄：環境變數輸入 (Context Input) ---
with st.sidebar:
    st.header("📝 環境參數設定")
    st.write("為了提高 AI 判讀準確度，請輸入當下狀況：")
    
    last_feed = st.slider("距離上一餐 (小時)", 0.0, 6.0, 2.5, 0.5)
    
    is_diaper_clean = st.radio(
        "目前的尿布狀態", 
        ["乾淨 (Clean)", "髒/濕 (Dirty/Wet)"], 
        index=0
    )

# --- 5. 錄音功能區 (核心功能) ---
st.subheader("1. 聲音採樣 (Audio Input)")

# 建立兩欄版面，讓按鈕不會佔滿整個螢幕寬度
col1, col2 = st.columns([1, 3])
with col1:
    st.write("操作指令：")
with col2:
    # 呼叫網頁錄音元件
    # 這是雲端版能運作的關鍵，它會調用手機/瀏覽器的麥克風
    audio = mic_recorder(
        start_prompt="🔴 點擊錄音 (Start)",
        stop_prompt="⬛ 停止並分析 (Stop)",
        key='recorder',
        format='wav'
    )

# --- 6. 分析與決策流程 ---
if audio:
    # 取得錄音的二進位資料
    audio_bytes = audio['bytes']
    
    st.success("✅ 錄音接收成功！AI 正在運算中...")
    
    # 將資料存為暫存檔，以便 librosa 讀取
    temp_filename = "cloud_upload.wav"
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)

    # 顯示播放器供確認
    st.audio(audio_bytes)

    try:
        with st.spinner('正在提取聲學特徵 (MFCC/RMS/BPM)...'):
            # A. 讀取音訊
            y, sr = librosa.load(temp_filename)
            
            # B. 提取關鍵特徵
            # 1. 能量強度 (Volume/RMS)
            rms = librosa.feature.rms(y=y)
            avg_volume = np.mean(rms)
            
            # 2. 音高頻率 (Pitch/Spectral Centroid)
            centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            avg_pitch_feature = np.mean(centroids)
            
            # 3. 節奏速度 (Tempo/BPM)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            bpm = tempo[0]

            # C. 視覺化儀表板 (Dashboard)
            st.subheader("2. 聲學特徵儀表板")
            
            # 繪製波形圖
            fig, ax = plt.subplots(figsize=(10, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color='#FF4B4B')
            ax.set_title("Waveform Analysis")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
            # 顯示數值指標
            m1, m2, m3 = st.columns(3)
            m1.metric("能量強度 (RMS)", f"{avg_volume:.3f}")
            m2.metric("音高頻率 (Hz)", f"{avg_pitch_feature:.0f}")
            m3.metric("節奏 (BPM)", f"{bpm:.0f}")

            # D. 邏輯決策樹 (Decision Tree)
            # 根據特徵數值進行分類
            predicted_type = "未知"
            urgency_color = "blue" # blue, orange, red, green
            
            if avg_volume > 0.08 and avg_pitch_feature > 2800:
                predicted_type = "疼痛 (Pain)"
                urgency_color = "red"
            elif bpm > 110 and avg_volume > 0.04:
                predicted_type = "飢餓 (Hunger)"
                urgency_color = "orange"
            elif avg_volume < 0.03:
                predicted_type = "疲倦 (Tired)"
                urgency_color = "blue"
            else:
                predicted_type = "不適/尋求關注"
                urgency_color = "green"

            # E. 輸出結果與建議
            st.subheader("3. 智慧決策建議 (SOP)")
            
            # 顯示判讀結果
            if urgency_color == "red":
                st.error(f"### 🔍 AI 判讀結果：{predicted_type}")
            elif urgency_color == "orange":
                st.warning(f"### 🔍 AI 判讀結果：{predicted_type}")
            elif urgency_color == "green":
                st.success(f"### 🔍 AI 判讀結果：{predicted_type}")
            else:
                st.info(f"### 🔍 AI 判讀結果：{predicted_type}")

            # 生成建議行動 (Action Plan)
            advice = ""
            
            if "疼痛" in predicted_type:
                advice = """
                🔴 **緊急處置 SOP：**
                1. **檢查外傷**：確認是否有頭髮纏繞手指/腳趾 (Hair tourniquet)。
                2. **量測體溫**：確認是否發燒。
                3. **觸診腹部**：若腹部緊繃可能是腸絞痛，請嘗試飛機抱或腹部按摩。
                > **注意**：若安撫無效且持續高頻尖叫，建議諮詢醫師。
                """
            elif "飢餓" in predicted_type:
                if last_feed < 1.5:
                    advice = """
                    🟡 **決策建議：**
                    * 距離上一餐時間較短，可能是 **口慾期 (討安撫)** 或 **需要拍嗝**。
                    * 建議先檢查是否有氣體未排出，或給予安撫奶嘴。
                    """
                else:
                    advice = """
                    🟢 **決策建議：**
                    * 生理時鐘與哭聲特徵吻合，判斷為 **飢餓**。
                    * 建議立即準備餵食。
                    """
            elif "疲倦" in predicted_type:
                advice = """
                🔵 **處置建議：**
                * 寶寶累過頭了 (Over-tired)。
                * **立即降低環境刺激**：關燈、關閉吵雜聲音。
                * 使用白噪音並進行包巾安撫，協助入睡。
                """
            else: # 不適或尋求關注
                if "髒" in is_diaper_clean:
                    advice = """
                    🟡 **優先行動：**
                    * 請優先 **更換尿布**。
                    * 檢查是否有尿布疹情形。
                    """
                else:
                    advice = """
                    🟢 **建議行動：**
                    * 生理需求似乎已滿足。
                    * 可能是 **無聊** 或 **太熱/太冷**。
                    * 建議變換抱姿，檢查後頸溫度，或與寶寶說話互動。
                    """

            st.markdown(advice)
            
            # 清除暫存檔，釋放空間
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    except Exception as e:
        st.error(f"分析過程發生錯誤：{e}")
        st.info("排除建議：請確認手機瀏覽器已授權使用麥克風，並嘗試錄製長一點的聲音 (3秒以上)。")

else:
    st.write("等待錄音... 請點擊上方按鈕開始。")
