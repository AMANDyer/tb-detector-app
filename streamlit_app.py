# streamlit_app.py - Fixed version with better errors
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="TB Detector", page_icon="🫁", layout="centered")
st.title("🫁 AI Tuberculosis Detector")
st.markdown("Upload a chest X-ray for instant analysis")

uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your X-ray", use_column_width=True)
    
    if st.button("Analyze Now"):
        with st.spinner("Sending to AI model..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                files = {"file": ("xray.jpg", uploaded_file.read(), "image/jpeg")}
                
                response = requests.post("http://127.0.0.1:8000/predict", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result["result"] == "NORMAL":
                        st.balloons()
                        st.success(f"✅ {result['message']}")
                    else:
                        st.error(f"⚠️ {result['message']}")
                    st.info(result["note"])
                else:
                    st.error(f"Backend Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Is FastAPI running on port 8000?")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")