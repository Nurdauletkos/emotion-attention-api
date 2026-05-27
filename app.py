"""
AI Emotion & Attention Analyzer
Дипломдық жұмыс: Мырзақұл Жаңыл
Л.Н. Гумилев атындағы Еуразия Ұлттық университеті
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from emotion_detector import EmotionDetector
from attention_detector import AttentionDetector

# --- БАПТАУЛАР ---
st.set_page_config(
    page_title="AI Emotion & Attention | Жаңыл Мырзақұл",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ДИЗАЙН ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #5b21b6 0%, #8b5cf6 100%); 
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.1); 
        padding: 10px 15px; 
        border-radius: 10px;
        margin: 5px 0; 
        transition: all 0.3s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.25); 
        transform: translateX(5px);
    }
    
    h1 { 
        color: #5b21b6; 
        font-weight: 800; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1); 
    }
    h2 { 
        color: #6d28d9; 
        border-bottom: 3px solid #8b5cf6; 
        padding-bottom: 10px; 
    }
    h3 { color: #7c3aed; }
    
    .feature-card {
        background: white; 
        padding: 25px; 
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1); 
        margin: 15px 0;
        transition: transform 0.3s; 
        border-left: 5px solid #8b5cf6;
    }
    .feature-card:hover { 
        transform: translateY(-5px); 
        box-shadow: 0 15px 40px rgba(0,0,0,0.15); 
    }
    
    .hero-block {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px; 
        border-radius: 25px; 
        color: white; 
        text-align: center;
        margin-bottom: 30px; 
        box-shadow: 0 15px 35px rgba(118,75,162,0.4);
    }
    .hero-block h1 { 
        color: white !important; 
        font-size: 42px; 
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3); 
    }
    
    .metric-card {
        background: white; 
        padding: 20px; 
        border-radius: 15px; 
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }
    .metric-value { 
        font-size: 36px; 
        font-weight: 800; 
        color: #5b21b6; 
    }
    .metric-label { 
        font-size: 14px; 
        color: #6b7280; 
        text-transform: uppercase; 
        letter-spacing: 1px; 
    }
    
    .author-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px; 
        border-radius: 20px; 
        color: white; 
        text-align: center;
        box-shadow: 0 15px 35px rgba(245,87,108,0.4);
    }
    .author-card h2 { color: white; border: none; }
    
    .info-box {
        background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid #7c3aed; 
        margin: 15px 0;
    }
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid #28a745; 
        margin: 15px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid #f59e0b; 
        margin: 15px 0;
    }
    .danger-box {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid #ef4444; 
        margin: 15px 0;
    }
    .emotion-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; 
        border-radius: 20px; 
        color: white; 
        text-align: center;
        box-shadow: 0 10px 30px rgba(118,75,162,0.4);
        margin: 15px 0;
    }
    .footer {
        text-align: center; 
        padding: 30px; 
        background: #5b21b6;
        color: white; 
        border-radius: 15px; 
        margin-top: 50px;
    }
    .back-btn {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(59,130,246,0.3);
    }
    .back-btn a {
        color: white !important;
        text-decoration: none;
        font-size: 18px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# --- ДЕТЕКТОРЛАРДЫ ЖҮКТЕУ ---
@st.cache_resource
def load_emotion_detector():
    return EmotionDetector(detector_backend='opencv')


@st.cache_resource
def load_attention_detector():
    return AttentionDetector(detector_backend='opencv')


# --- БҮЙІРЛІК МЕНЮ ---
with st.sidebar:
    st.markdown("# 😊 Emotion & Attention AI")
    st.markdown("---")
    
    page = st.radio(
        "📍 **Бөлімді таңдаңыз:**",
        ["🏠 Басты бет",
         "😊 Эмоция Анықтау",
         "👁 Назар Бақылау",
         "📚 Қалай жұмыс істейді?",
         "👨‍💻 Автор туралы"]
    )
    
    st.markdown("---")
    st.markdown("### 🔗 Басқа жобам")
    st.markdown("""
    <div style="background:rgba(255,255,255,0.15); padding:15px; border-radius:10px;">
    <p style="margin:0; font-size:14px;">👤 <b>Жас пен жынысты</b><br>анықтайтын жоба</p>
    <a href="https://zhanyl-age-gender.streamlit.app" target="_blank" 
       style="display:inline-block; margin-top:10px; padding:8px 15px; 
       background:white; color:#5b21b6 !important; border-radius:8px; 
       text-decoration:none; font-weight:700; font-size:13px;">
       Көру →
    </a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Жоба туралы")
    st.markdown("""
    Бұл жоба жасанды интеллект 
    көмегімен **эмоция мен назарды** 
    анықтайды.
    
    🎓 Мектеп оқушыларына 
    арналған дипломдық жоба.
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Дипломдық жұмыс")
    st.markdown("""
    **Автор:** Мырзақұл Жаңыл  
    **ЖОО:** Л.Н. Гумилев атындағы  
    Еуразия Ұлттық университеті
    """)


# 🏠 БАСТЫ БЕТ
if page == "🏠 Басты бет":
    st.markdown("""
    <div class="hero-block">
        <h1>😊 Emotion & Attention AI</h1>
        <h3 style="color:white; opacity:0.95;">Жасанды интеллектпен эмоция мен назарды анықтау</h3>
        <p style="font-size:18px; margin-top:20px;">
            🎓 Мектеп оқушыларына арналған білім беретін жоба
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🚀 Жүйенің негізгі мүмкіндіктері")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>😊 Эмоция Анықтау</h3>
            <p>7 негізгі эмоцияны нақты уақытта анықтайды:</p>
            <p>😠 Ашулы · 🤢 Жиіркеніш · 😨 Қорқыныш · 😊 Бақытты · 
            😢 Мұңды · 😮 Таңқалу · 😐 Бейтарап</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>👁 Назар Бақылау</h3>
            <p>Пайдаланушының бас бағытын талдау арқылы 
            назарын real-time қадағалайды.</p>
            <p>🟢 Назарда · 🟡 Алаңдау · 🔴 Назар жоғалды</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 📈 Жоба статистикасы")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">7</div><div class="metric-label">Эмоция түрі</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Назар деңгейі</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">90%+</div><div class="metric-label">Дәлдік</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">Real-time</div><div class="metric-label">Жұмыс</div></div>', unsafe_allow_html=True)
    
    st.markdown("## 📌 Пайдалану нұсқаулығы")
    st.markdown("""
    <div class="info-box">
        <ol style="line-height:2; font-size:16px;">
            <li><b>Бүйірлік менюден</b> бөлімді таңдаңыз: Эмоция немесе Назар</li>
            <li><b>Режимді таңдаңыз:</b> Камера немесе сурет жүктеу</li>
            <li><b>Бетіңізді көрсетіңіз</b> немесе суретті жүктеңіз</li>
            <li><b>Нәтиже</b> экранда көрсетіледі</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="success-box">
        <h3>💡 Білдіңіз бе?</h3>
        <p>Адамның бет әлпеті өте көп ақпарат береді — қуанышын, мұңын, ашуын. 
        Жасанды интеллект мыңдаған суреттен үйренеді, сосын сіздің эмоцияңызды 
        дөл анықтайды! Назар бақылау оқу процесін жақсартуға, ұйқысыздық пен 
        алаңдауды табуға көмектеседі.</p>
    </div>
    """, unsafe_allow_html=True)


# 😊 ЭМОЦИЯ
elif page == "😊 Эмоция Анықтау":
    st.markdown("# 😊 Эмоция Анықтау")
    
    st.markdown("""
    <div class="info-box">
        <h3>🤖 Бет әлпеті бойынша эмоция анықтау</h3>
        <p>Жүйе сіздің бетіңізді табады, нейрон желі арқылы талдау жасайды және 
        7 эмоцияның қайсысы басым екенін көрсетеді.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔄 Модель жүктелуде..."):
        detector = load_emotion_detector()
    
    st.success("✅ Модель дайын!")
    
    mode = st.radio(
        "🎯 **Режимді таңдаңыз:**",
        ["📷 Камерадан сурет", "🖼️ Сурет жүктеу"],
        horizontal=True
    )
    
    st.markdown("---")
    
    def analyze_emotion(image_input):
        """Сурет талдау"""
        if isinstance(image_input, np.ndarray):
            img_array = image_input
        else:
            img = Image.open(image_input)
            img_array = np.array(img)
        
        # RGB → BGR
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        with st.spinner("🔍 Талдау жүруде..."):
            results = detector.detect_from_image(img_bgr)
        
        if not results:
            st.warning("⚠️ Бет табылмады. Бет анық көрінетін суретті жүктеңіз.")
            st.image(img_array, caption="Жүктелген сурет", use_container_width=True)
            return
        
        # Рамка салу
        result_img = img_bgr.copy()
        for face in results:
            region = face.get('region', {})
            if region:
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 0)
                h = region.get('h', 0)
                if w > 0 and h > 0:
                    cv2.rectangle(result_img, (x, y), (x + w, y + h), (139, 92, 246), 4)
        
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="✅ Талдау нәтижесі", use_container_width=True)
        
        # Әр бет үшін нәтиже
        for i, face in enumerate(results, 1):
            if len(results) > 1:
                st.markdown(f"### 👤 Бет #{i}")
            
            st.markdown(f"""
            <div class="emotion-result">
                <div style="font-size:72px; line-height:1;">{face['emoji']}</div>
                <h2 style="color:white; border:none; font-size:36px;">{face['dominant_emotion_kz']}</h2>
                <p style="font-size:18px; opacity:0.95;">
                    {face['intensity_label']} · {face['intensity']:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📊 Барлық эмоциялар")
            
            # Эмоция тізімі
            for emo in face['all_emotions']:
                score = emo['score']
                color = "#8b5cf6" if emo['emotion'] == face['dominant_emotion'] else "#e5e7eb"
                st.markdown(f"""
                <div style="background:white; padding:12px; border-radius:10px; 
                margin:8px 0; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
                    <div style="display:flex; align-items:center; gap:15px;">
                        <div style="font-size:28px;">{emo['emoji']}</div>
                        <div style="flex:1;">
                            <div style="font-weight:700; color:#1f2937; margin-bottom:4px;">
                                {emo['emotion_kz']}
                            </div>
                            <div style="background:#e5e7eb; height:8px; border-radius:4px; overflow:hidden;">
                                <div style="background:{color}; width:{score}%; 
                                height:100%; border-radius:4px;"></div>
                            </div>
                        </div>
                        <div style="font-weight:700; color:#5b21b6; font-family:monospace; min-width:60px; text-align:right;">
                            {score:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    if mode == "📷 Камерадан сурет":
        st.markdown("### 📷 Камерадан сурет түсіру")
        st.markdown("""
        <div class="info-box">
            <p>📌 Төмендегі камера батырмасын басып, бетіңізді көрсетіңіз. 
            Сурет автоматты түрде талданады.</p>
        </div>
        """, unsafe_allow_html=True)
        
        camera_image = st.camera_input("📸 Сурет түсіру")
        if camera_image is not None:
            analyze_emotion(camera_image)
    
    else:
        st.markdown("### 🖼️ Сурет жүктеу")
        uploaded_file = st.file_uploader(
            "Бет көрінетін суретті жүктеңіз",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            analyze_emotion(uploaded_file)


# 👁 НАЗАР
elif page == "👁 Назар Бақылау":
    st.markdown("# 👁 Назар Бақылау")
    
    st.markdown("""
    <div class="info-box">
        <h3>🤖 Пайдаланушының назарын анықтау</h3>
        <p>Жүйе сіздің бас бағытыңызды талдау арқылы назарыңызды анықтайды:</p>
        <ul>
            <li>🟢 <b>Назарда</b> — камераға қарап тұрсыз</li>
            <li>🟡 <b>Алаңдау</b> — сәл басқа жаққа қарап тұрсыз</li>
            <li>🔴 <b>Назар жоғалды</b> — басқа жаққа бұрылдыңыз</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔄 Модель жүктелуде..."):
        detector = load_attention_detector()
    
    st.success("✅ Модель дайын!")
    
    mode = st.radio(
        "🎯 **Режимді таңдаңыз:**",
        ["📷 Камерадан сурет", "🖼️ Сурет жүктеу"],
        horizontal=True
    )
    
    st.markdown("---")
    
    def analyze_attention(image_input):
        """Назарды талдау"""
        if isinstance(image_input, np.ndarray):
            img_array = image_input
        else:
            img = Image.open(image_input)
            img_array = np.array(img)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_array
        
        with st.spinner("🔍 Талдау жүруде..."):
            result = detector.detect_from_image(img_bgr)
        
        if not result.get('face_detected'):
            st.warning("⚠️ Бет табылмады. Бет анық көрінетін суретті жүктеңіз.")
            st.image(img_array, caption="Жүктелген сурет", use_container_width=True)
            return
        
        attention = result['attention']
        status = attention['status']
        focus_score = attention['focus_score']
        region = result.get('region', {})
        
        # Түс пен emoji
        if status == 'focused':
            color_bgr = (40, 167, 69)  # жасыл
            emoji = "🟢"
            box_class = "success-box"
        elif status == 'distracted':
            color_bgr = (245, 158, 11)  # сары
            emoji = "🟡"
            box_class = "warning-box"
        else:
            color_bgr = (239, 68, 68)  # қызыл
            emoji = "🔴"
            box_class = "danger-box"
        
        # Рамка салу
        result_img = img_bgr.copy()
        if region:
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)
            if w > 0 and h > 0:
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color_bgr, 4)
        
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="✅ Талдау нәтижесі", use_container_width=True)
        
        # Негізгі нәтиже
        st.markdown(f"""
        <div class="{box_class}">
            <div style="text-align:center;">
                <div style="font-size:72px;">{emoji}</div>
                <h2 style="border:none; margin:10px 0;">{attention['status_kz']}</h2>
                <p style="font-size:16px;">{attention['message']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Метрикалар
        st.markdown("### 📊 Толық ақпарат")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{focus_score:.0f}%</div><div class="metric-label">Фокус ұпайы</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{attention["direction_kz"]}</div><div class="metric-label">Бағыты</div></div>', unsafe_allow_html=True)
        with col3:
            confidence = result.get('confidence', 0) * 100
            st.markdown(f'<div class="metric-card"><div class="metric-value">{confidence:.0f}%</div><div class="metric-label">Сенімділік</div></div>', unsafe_allow_html=True)
        
        # Прогресс бар
        st.markdown("### 🎯 Фокус деңгейі")
        st.progress(focus_score / 100)
    
    if mode == "📷 Камерадан сурет":
        st.markdown("### 📷 Камерадан сурет түсіру")
        st.markdown("""
        <div class="info-box">
            <p>📌 Камераға тура қарап, сурет түсіріңіз. Жүйе сіздің назарыңыздың 
            қайда екенін анықтайды.</p>
        </div>
        """, unsafe_allow_html=True)
        
        camera_image = st.camera_input("📸 Сурет түсіру")
        if camera_image is not None:
            analyze_attention(camera_image)
    
    else:
        st.markdown("### 🖼️ Сурет жүктеу")
        uploaded_file = st.file_uploader(
            "Бет көрінетін суретті жүктеңіз",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            analyze_attention(uploaded_file)


# 📚 ҚАЛАЙ ЖҰМЫС ІСТЕЙДІ
elif page == "📚 Қалай жұмыс істейді?":
    st.markdown("# 📚 Қалай жұмыс істейді?")
    
    st.markdown("""
    <div class="hero-block" style="padding:25px;">
        <h2 style="color:white;">🧠 Жасанды интеллект — бұл сиқыр емес, ғылым!</h2>
        <p style="font-size:18px;">Кел, эмоция мен назар анықтау қалай жұмыс істейтінін үйренейік!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 😊 Эмоция анықтау")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Қалай жұмыс істейді?</h3>
            <ol>
                <li>Камера сурет түсіреді</li>
                <li>AI бетіңізді табады</li>
                <li>Бет әлпетіңізді талдайды</li>
                <li>7 эмоцияның ықтималдығын есептейді</li>
                <li>Ең басымын көрсетеді</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 7 негізгі эмоция</h3>
            <p>😊 <b>Бақытты</b> — қуанышты</p>
            <p>😢 <b>Мұңды</b> — көңілсіз</p>
            <p>😠 <b>Ашулы</b> — ренжіген</p>
            <p>😮 <b>Таңқалу</b> — таңданған</p>
            <p>😨 <b>Қорқыныш</b> — қорыққан</p>
            <p>🤢 <b>Жиіркеніш</b></p>
            <p>😐 <b>Бейтарап</b> — қалыпты</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 👁 Назар бақылау")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Қалай анықтайды?</h3>
            <ol>
                <li>Бет аумағын табады</li>
                <li>Көздердің координаталарын алады</li>
                <li>Бас бұрылысын есептейді (yaw, pitch)</li>
                <li>Назар деңгейін шығарады</li>
            </ol>
            <p>💡 Сіз бір нүктеге қарап тұрсаңыз — назарда. 
            Басқа жаққа қарасаңыз — алаңдау.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🚦 3 деңгей</h3>
            <p>🟢 <b>Назарда</b> (Focused)<br>
            Камераға тура қарап тұрсыз. Фокус 80-100%.</p>
            <p>🟡 <b>Алаңдау</b> (Distracted)<br>
            Сәл басқа жаққа бұрылдыңыз. Фокус 50-80%.</p>
            <p>🔴 <b>Назар жоғалды</b> (Lost)<br>
            Басқа жерге қарап тұрсыз. Фокус 0-50%.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🔬 Қандай технология қолданылды?")
    
    st.markdown("""
    <div class="success-box">
        <h3>📊 Техникалық бөлік:</h3>
        <ul>
            <li><b>🤖 DeepFace</b> — Facebook жасаған кітапхана, бет талдау үшін</li>
            <li><b>🧠 Нейрон желі:</b> VGG-Face, FaceNet, OpenFace (алдын ала оқытылған)</li>
            <li><b>⚙️ TensorFlow</b> — Google-дің машиналық оқыту фреймворкі</li>
            <li><b>📷 OpenCV</b> — суреттерді өңдеуге арналған құрал</li>
            <li><b>🐍 Python + Streamlit</b> — веб интерфейс үшін</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🌟 Бұл қайда қолданылады?")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>📚 Білім беру</h3><p>Оқушының назарын қадағалау</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🚗 Көлік</h3><p>Жүргізуші ұйқылы ма тексеру</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🎮 Ойындар</h3><p>Эмоцияға бейімделген ойындар</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>🏥 Медицина</h3><p>Депрессия мен стрессті табу</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🛍 Маркетинг</h3><p>Жарнамаға реакцияны өлшеу</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>🤖 Робототехника</h3><p>Адамға бейімделген роботтар</p></div>', unsafe_allow_html=True)


# 👨‍💻 АВТОР
elif page == "👨‍💻 Автор туралы":
    st.markdown("# 👨‍💻 Автор туралы")
    
    st.markdown("""
    <div class="author-card">
        <h1 style="color:white; font-size:48px;">🎓</h1>
        <h2 style="color:white;">Мырзақұл Жаңыл</h2>
        <h3 style="color:white; opacity:0.9;">Бакалавр студенті</h3>
        <p style="font-size:18px; margin-top:20px;">
            🏛 Л.Н. Гумилев атындағы Еуразия Ұлттық университеті
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 📋 Жоба туралы мәлімет")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Жобаның мақсаты</h3>
            <p>Жасанды интеллект көмегімен <b>эмоция мен назарды</b> анықтайтын 
            практикалық жүйе жасау. Мектеп оқушыларына AI технологиясын 
            <b>көрнекі түрде</b> түсіндіру.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🌟 Жобаның маңызы</h3>
            <p>Эмоция мен назар бақылау — <b>білім беру, медицина, көлік</b> 
            салаларында өзекті. Бұл жоба <b>қазақ тілінде</b> AI білім беруге 
            үлес қосады.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🔗 Барлық жобаларым")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="feature-card" style="border-left-color:#3b82f6;">
            <h3>👤 Жас пен жыныс анықтау</h3>
            <p>PyTorch-та өзім оқытқан модель. UTKFace + IMDB-Wiki датасеттері.</p>
            <a href="https://zhanyl-age-gender.streamlit.app" target="_blank"
               style="display:inline-block; margin-top:10px; padding:10px 20px;
               background:#3b82f6; color:white; border-radius:8px;
               text-decoration:none; font-weight:700;">
               Сайтты ашу →
            </a>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card" style="border-left-color:#8b5cf6;">
            <h3>😊 Эмоция + Назар (бұл жоба)</h3>
            <p>DeepFace кітапханасын қолдану. 7 эмоция + 3 назар деңгейі.</p>
            <div style="display:inline-block; margin-top:10px; padding:10px 20px;
               background:#8b5cf6; color:white; border-radius:8px;
               font-weight:700;">
               Қазір осындасыз
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## 🛠 Қолданылған технологиялар")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>🐍 Python</h3><p>Негізгі тіл</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🤖 DeepFace</h3><p>Бет талдау</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>📷 OpenCV</h3><p>Сурет өңдеу</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>🎨 Streamlit</h3><p>Веб интерфейс</p></div>', unsafe_allow_html=True)
    
    st.markdown("## 📅 Жоба этаптары")
    st.markdown("""
    <div class="feature-card">
        <ol style="font-size:16px; line-height:2;">
            <li>📚 <b>Зерттеу:</b> Эмоция мен назар бақылау тақырыбын зерттеу</li>
            <li>🔬 <b>Технология таңдау:</b> DeepFace кітапханасы</li>
            <li>🏗 <b>Модуль құру:</b> emotion_detector + attention_detector</li>
            <li>⚙️ <b>API жасау:</b> FastAPI арқылы backend</li>
            <li>🎨 <b>Интерфейс:</b> Streamlit арқылы веб</li>
            <li>🧪 <b>Тестілеу:</b> Әртүрлі суреттер мен жағдайларды тексеру</li>
            <li>🚀 <b>Шығару:</b> Streamlit Cloud-та орналастыру</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🙏 Алғыс білдіру")
    st.markdown("""
    <div class="success-box">
        <p style="font-size:16px;">
        Бұл жобаны жасауға көмектескен <b>ғылыми жетекшіме</b>, <b>отбасыма</b> 
        және <b>достарыма</b> алғысымды білдіремін. Сондай-ақ <b>ашық кодты</b> 
        жасап жариялаған бағдарламашыларға ризалығымды білдіремін.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>🎓 Дипломдық жұмыс © 2025</h3>
        <p>Мырзақұл Жаңыл · Л.Н. Гумилев атындағы Еуразия Ұлттық университеті</p>
    </div>
    """, unsafe_allow_html=True)
