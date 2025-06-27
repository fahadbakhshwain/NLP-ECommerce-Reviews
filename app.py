import streamlit as st
import pandas as pd
import joblib
import re # لاستخدام التعابير النمطية (Regular Expressions)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # قد لا تحتاجها إذا لم تستخدم Stemming

# تنزيل بيانات NLTK الضرورية (مرة واحدة فقط)
# يجب التأكد من أن هذه البيانات متاحة على خادم Streamlit
# في بعض الأحيان قد تحتاج إلى إضافة هذه الأسطر في بداية app.py لضمان التنزيل
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    st.info("تم تنزيل NLTK stopwords.")

# --- تعريف دالة معالجة النصوص (Preprocessing Function) ---
# هذه الدالة يجب أن تكون مطابقة تماماً للدالة التي استخدمتها في Notebook لتدريب الموديل
def preprocess_text(text):
    # 1. تحويل النص إلى حروف صغيرة
    text = text.lower()
    
    # 2. إزالة علامات الترقيم والأرقام (الاحتفاظ فقط بالأحرف الأبجدية والمسافات)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. تقسيم النص إلى كلمات
    tokens = text.split()
    
    # 4. إزالة الكلمات المتوقفة (Stopwords)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Stemming (اختياري، إذا كنت قد استخدمته في تدريب الموديل)
    # stemmer = PorterStemmer()
    # tokens = [stemmer.stem(word) for word in tokens]
    
    # 6. دمج الكلمات مرة أخرى في نص واحد
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="تطبيق تحليل آراء العملاء (NLP)",
    page_icon="💬",
    layout="centered"
)

# --- تحميل أداة تحويل النصوص (Vectorizer) وموديل تحليل المشاعر ---
try:
    # تأكد أن هذه الأسماء والمسارات مطابقة تماماً لملفاتك المحفوظة
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    sentiment_model = joblib.load('models/sentiment_classifier_model.joblib')
except FileNotFoundError:
    st.error("ملفات الموديل أو Vectorizer غير موجودة! يرجى التأكد من وجود 'tfidf_vectorizer.joblib' و 'sentiment_classifier_model.joblib' في مجلد 'models'.")
    st.stop()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل الموديلات: {e}")
    st.stop()

# --- عنوان التطبيق والوصف ---
st.title('💬 تطبيق تحليل آراء العملاء (NLP)')
st.write("أدخل مراجعة العميل أو تعليقه هنا لتحليل مشاعره واستخلاص الأفكار الرئيسية.")
st.write("---")

# --- قسم إدخال المستخدم ---
user_review = st.text_area("أدخل مراجعة العميل هنا:", height=150)

# --- زر التحليل ومنطق التنبؤ ---
if st.button("تحليل المراجعة", type="primary"):
    if user_review:
        # 1. معالجة النص المدخل
        cleaned_review = preprocess_text(user_review)
        
        # 2. تحويل النص المعالج إلى ميزات رقمية باستخدام Vectorizer
        # ملاحظة: يجب أن نمرر قائمة (list) حتى لو كانت تحتوي على نص واحد
        vectorized_review = vectorizer.transform([cleaned_review])
        
        # 3. إجراء التنبؤ بالمشاعر باستخدام الموديل
        prediction = sentiment_model.predict(vectorized_review)
        
        # 4. الحصول على احتمالية كل فئة (اختياري، لكنه يعطي رؤية أفضل)
        prediction_proba = sentiment_model.predict_proba(vectorized_review)
        
        # 5. تفسير النتيجة وعرضها
        sentiment_labels = ['سلبية', 'محايدة', 'إيجابية'] # يجب أن تتطابق مع ترتيب الفئات في بياناتك
        predicted_sentiment = sentiment_labels[prediction[0]]
        
        st.write("---")
        st.subheader("نتائج التحليل:")
        
        if predicted_sentiment == 'إيجابية':
            st.success(f"المشاعر المتوقعة: **{predicted_sentiment}** 🎉")
        elif predicted_sentiment == 'سلبية':
            st.error(f"المشاعر المتوقعة: **{predicted_sentiment}** 😠")
        else: # محايدة
            st.info(f"المشاعر المتوقعة: **{predicted_sentiment}** 😐")
        
        # عرض الاحتماليات لكل فئة
        st.write(f"احتمالية السلبية: {prediction_proba[0][0]:.2%}")
        st.write(f"احتمالية المحايدة: {prediction_proba[0][1]:.2%}")
        st.write(f"احتمالية الإيجابية: {prediction_proba[0][2]:.2%}")

        # يمكن هنا إضافة جزء لاستخراج الكلمات المفتاحية إذا كان لديك موديل لذلك
        # st.subheader("الكلمات المفتاحية/المواضيع الرئيسية:")
        # st.write("... (سيتم عرضها هنا)")

    else:
        st.warning("الرجاء إدخال نص لتحليله.", icon="⚠️")