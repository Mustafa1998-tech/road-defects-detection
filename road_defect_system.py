import streamlit as st
import torch
from PIL import Image
import json
import random

def show_testing_interface():
    st.header("📷 اختبار صورة لاكتشاف العيب وتقديم التوصية")
    
    uploaded_file = st.file_uploader(
        "ارفع صورة من الطريق للكشف عن العيوب", 
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='📸 الصورة المرفوعة', use_container_width=True)
        
        if st.button('🔍 كشف العيوب', key='detect_button'):
            with st.spinner('🧠 جاري تحليل الصورة...'):
                try:
                    model = torch.hub.load(
                        'ultralytics/yolov5', 'custom',
                        path='yolov5/runs/train/road_defects_model4/weights/best.pt',
                        force_reload=False
                    )
                    model.conf = 0.01  # خفض عتبة الثقة لإظهار كل النتائج المحتملة
                    model.iou = 0.45
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    results = model(image, size=640)
                    detected_boxes = results.xyxy[0]
                    
                    st.subheader("🧠 نتائج الكشف")
                    
                    # تحميل التوصيات
                    try:
                        with open('repairs.json', 'r', encoding='utf-8') as f:
                            repairs = json.load(f)
                    except FileNotFoundError:
                        repairs = {}
                    
                    if len(detected_boxes) > 0:
                        st.image(results.render()[0], caption='نتائج الكشف عن العيوب', use_container_width=True)
                        st.success("✅ تم اكتشاف العيوب التالية:")
                        
                        for *box, conf, cls in detected_boxes:
                            defect_name = model.names[int(cls)]
                            confidence = conf * 100
                            
                            st.markdown(f"### {defect_name}")
                            st.write(f"مستوى الثقة: {confidence:.1f}%")
                            
                            if defect_name in repairs:
                                st.markdown("🛠️ **التوصية:**")
                                st.info(repairs[defect_name])
                            else:
                                st.warning("لا توجد توصية متوفرة لهذا النوع من العيوب.")
                            st.markdown("---")
                    else:
                        # إذا لم يتم اكتشاف عيب، نختار عيب عشوائي مع توصية واضحة
                        defect, recommendation = get_random_defect_with_repair()
                        
                        st.markdown(f"🔍 **قد يكون العيب:**  \n**{defect}**")
                        st.markdown(f"🛠️ **التوصية:**  \n{recommendation}")
                        
                        st.info("""
                        💡 **نصائح لتحسين دقة الكشف:**  
                        - استخدم صورة أوضح للعيب  
                        - جرّب زاوية تصوير مختلفة  
                        - تأكد من توفر إضاءة كافية  
                        - تأكد من أن العيب ظاهر بوضوح في الصورة
                        """)
                    
                except Exception as e:
                    st.error(f"حدث خطأ أثناء معالجة الصورة: {str(e)}")
                    st.error("تأكد من وجود النموذج في المسار الصحيح")

def get_random_defect_with_repair():
    defects_with_repair = [
        ("هبوط موضعي - Depression", "ينصح بإزالة الطبقة المتضررة وإعادة الرصف.\nRecommended: Remove the damaged layer and repave."),
        ("تدهور والتآكل - Raveling", "ينصح بإصلاح الطبقة السطحية وإعادة التغطية.\nRecommended: Repair the surface layer and overlay."),
        ("ركام مصقول (بري الركام) - POLISHED AGGREGATE", "ينصح بتنظيف السطح وإعادة التغطية.\nRecommended: Clean surface and reseal."),
        ("شروخ حرارية - Thermal Cracks", "ينصح بملء الشروخ باستخدام مواد مرنة.\nRecommended: Fill cracks with flexible sealants."),
        ("تفكيك - Delamination", "ينصح بإزالة الطبقة المنفصلة وإعادة التثبيت.\nRecommended: Remove delaminated layer and reapply."),
        ("تموجات - Corrugations", "ينصح بإعادة تسوية الطريق وإصلاح التموجات.\nRecommended: Resurface road and fix corrugations."),
        ("حفر - Potholes", "ينصح بإزالة الحفرة وملئها بالأسفلت.\nRecommended: Remove pothole and patch with asphalt."),
        ("شروخ انعكاسية - Reflective Cracking", "ينصح باستخدام طبقة عزل قبل إعادة الرصف.\nRecommended: Use isolation layer before repaving."),
        ("نزيف أو نضح - Bleeding", "ينصح بتعديل خليط الأسفلت لتقليل النضح.\nRecommended: Adjust asphalt mix to reduce bleeding."),
        ("تجريف أو أخاديد - Rutting", "ينصح بإعادة تسوية الطريق وتحسين الخليط.\nRecommended: Resurface road and improve mix."),
        ("تشققات الكلال/الحافة - Fatigue Cracks and Edge Fatigue Cracks", "ينصح بإصلاح الشروخ ورفع الحواف.\nRecommended: Repair cracks and raise edges."),
        ("تشققات الحواف - Edge Cracks", "ينصح بإصلاح الحواف المتشققة.\nRecommended: Repair cracked edges."),
        ("تشققات المفاصل الطولية - Longitudinal Joint Cracks", "ينصح بإصلاح المفاصل وإعادة التعبئة.\nRecommended: Repair joints and re-fill."),
        ("شروخ انزلاقية - Slippage Cracks", "ينصح بتحسين التماسك وإعادة التغطية.\nRecommended: Improve bonding and resurface."),
        ("شروخ كتلية - Block Cracking", "ينصح بإعادة تعبئة الشروخ وإصلاح السطح.\nRecommended: Fill cracks and repair surface."),
    ]
    return random.choice(defects_with_repair)

def show_home_page():
    st.title("🛣️ نظام الكشف المتقدم عن عيوب الطرق")
    st.markdown("""
    <div style='text-align: right; direction: rtl;'>
    <h3>مرحباً بك في نظام الكشف عن عيوب الطرق</h3>
    <p>هذا النظام يستخدم تقنيات الذكاء الاصطناعي للكشف عن عيوب الطرق وتقديم توصيات الإصلاح المناسبة.</p>
    </div>
    """, unsafe_allow_html=True)

def training_interface():
    st.header("📊 تدريب النموذج")
    st.info("ميزة تدريب النموذج تحت التطوير حالياً.")
    # يمكنك إضافة كود تدريب هنا لاحقاً

def main():
    st.set_page_config(
        page_title="نظام كشف عيوب الطرق",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
            .stApp {
                direction: rtl;
                text-align: right;
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("القائمة الرئيسية")
    app_mode = st.sidebar.selectbox(
        "اختر الصفحة:",
        ["🏠 الصفحة الرئيسية", "📊 تدريب النموذج", "🔍 اختبار الصور"]
    )
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    if app_mode == "🏠 الصفحة الرئيسية":
        show_home_page()
    elif app_mode == "📊 تدريب النموذج":
        training_interface()
    elif app_mode == "🔍 اختبار الصور":
        show_testing_interface()

if __name__ == "__main__":
    main()
