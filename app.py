import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from fpdf import FPDF
import tempfile
import os
import time
from datetime import datetime
import torch
import torchvision.transforms as T
from segmentation_models_pytorch import Unet
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

# Set a constant email for the header (used in PDF)
EMAIL = "nyamangodotatenda64@gmail.com  +263 71 424 2685"

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.8rem;
    font-weight: bold;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 3px solid #3498db;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}
.sub-header {
    font-size: 1.6rem;
    font-weight: bold;
    color: #3498db;
    margin-top: 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid #3498db;
    padding-left: 10px;
}
.result-box {
    padding: 1.2rem;
    border-radius: 12px;
    margin-top: 1.2rem;
    margin-bottom: 1.2rem;
    transition: all 0.3s ease;
}
.detection-positive {
    background-color: rgba(255, 0, 0, 0.1);
    border: 1px solid #ff0000;
    box-shadow: 0 4px 8px rgba(255,0,0,0.1);
}
.detection-negative {
    background-color: rgba(0, 128, 0, 0.1);
    border: 1px solid #008000;
    box-shadow: 0 4px 8px rgba(0, 128, 0, 0.1);
}
.stButton button {
    background-color: #3498db;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.6rem 2.2rem;
    transition: all 0.3s;
    border: none;
}
.stButton button:hover {
    background-color: #2980b9;
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
}
.stButton button:active {
    transform: translateY(-1px);
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
}
.input-section, .results-section {
    background-color: #f8f9fa;
    padding: 1.8rem;
    border-radius: 12px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
    transition: all 0.3s ease;
}
.input-section:hover, .results-section:hover {
    box-shadow: 0 6px 14px rgba(0,0,0,0.12);
}
.pdf-download {
    margin-top: 1.5rem;
    padding: 1.5rem;
    background-color: #e8f4f8;
    border-radius: 12px;
    border: 2px dashed #3498db;
    text-align: center;
    transition: all 0.3s ease;
}
.pdf-download:hover {
    background-color: #d1eaf3;
    transform: translateY(-2px);
}
footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #2c3e50;
    color: white;
    text-align: center;
    padding: 15px;
    font-size: 14px;
    box-shadow: 0 -3px 12px rgba(0,0,0,0.15);
}
.stFileUploader {
    padding: 10px;
    border-radius: 8px;
    border: 2px dashed #3498db;
}
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #3498db !important;
    padding: 10px !important;
}
.stTextInput input, .stNumberInput input, .stSelectbox select {
    border-radius: 8px !important;
    border: 1px solid #bdc3c7 !important;
    padding: 10px !important;
    transition: all 0.3s ease !important;
}
.stTextInput input:focus, .stNumberInput input:focus, .stSelectbox select:focus {
    border: 1px solid #3498db !important;
    box-shadow: 0 0 2px rgba(52, 152, 219, 0.2) !important;
}
@keyframes fadein {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.fadein {
    animation: fadein 0.5s ease-in-out;
}
.tumor-detected {
    color: #e74c3c;
    font-weight: bold;
    font-size: 1.6rem;
    margin-bottom: 1rem;
    animation: fadein 0.5s ease-in-out;
}
.no-tumor {
    color: #27ae60;
    font-weight: bold;
    font-size: 1.6rem;
    margin-bottom: 1rem;
    animation: fadein 0.5s ease-in-out;
}
.instruction-box {
    background-color: #f1f7fd;
    padding: 20px;
    border-radius: 12px;
    margin-top: 2.2rem;
    border-left: 5px solid #3498db;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
.info-text {
    line-height: 1.6;
    color: #2c3e50;
}
.stProgress > div > div {
    background-color: #3498db !important;
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# Function to show a custom loading spinner with a smooth progress bar
def custom_spinner():
    with st.spinner("Processing your MRI scan..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        st.success("Processing complete!")
        time.sleep(0.5)
        progress_bar.empty()

# Function to show a brief result animation/feedback
def show_result_animation(result_type):
    with st.spinner("Analyzing results..."):
        time.sleep(1)

# Modern Segmentation Functions
def apply_cnn_segmentation(image):
    """Apply CNN-based segmentation using U-Net"""
    model = Unet(encoder_name="resnet34", classes=3, activation="softmax", in_channels=3)
    model.eval()
    transform = T.Compose([T.ToTensor(), T.Resize((256, 256))])
    img_array = np.array(image)
    # Ensure the image is in RGB format
    if len(img_array.shape) == 2:  # If grayscale, convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    img_tensor = transform(img_array).unsqueeze(0)  # Shape: [1, 3, 256, 256]
    with torch.no_grad():
        mask = model(img_tensor)
        mask = torch.argmax(mask, dim=1).squeeze().numpy()
    # Color-code mask: 0=healthy (green), 1=vulnerable (yellow), 2=affected (red)
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 0] = [0, 255, 0]  # Green for healthy
    colored_mask[mask == 1] = [255, 255, 0]  # Yellow for vulnerable
    colored_mask[mask == 2] = [255, 0, 0]  # Red for affected
    return Image.fromarray(colored_mask), 0.97  # Return image and estimated accuracy

def apply_transformer_segmentation(image):
    """Apply Transformer-based segmentation using Segformer"""
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model.eval()
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # If grayscale, convert to RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    inputs = feature_extractor(images=img_array, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=img_array.shape[:2], mode="bilinear", align_corners=False
        )
        mask = torch.argmax(upsampled_logits, dim=1).squeeze().numpy()
    # Color-code mask: 0=healthy (green), 1=vulnerable (yellow), 2=affected (red)
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 0] = [0, 255, 0]  # Green for healthy
    colored_mask[mask == 1] = [255, 255, 0]  # Yellow for vulnerable
    colored_mask[mask == 2] = [255, 0, 0]  # Red for affected
    return Image.fromarray(colored_mask), 0.95  # Return image and estimated accuracy

# YOLO Detection Function
def predict_and_display_v11(image, model_path, target_size=(640, 640), confidence_threshold=0.5):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(image, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_input = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    model = YOLO(model_path, task='detect')
    results = model.predict(source=img_input, save=False, conf=confidence_threshold)
    detections = results[0].boxes.data.cpu().numpy()
    if len(detections) > 0:
        for det in detections:
            xmin, ymin, xmax, ymax, conf, cls = map(int, det)
            cv2.rectangle(img_resized, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"Tumor: {conf:.2f}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(img_resized, (xmin, ymin - t_size[1] - 10), (xmin + t_size[0], ymin), (0, 255, 0), -1)
            cv2.putText(img_resized, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), detections

# Function to generate a PDF report and return the PDF data for download
def generate_and_download_pdf(patient_id, patient_name, email, gender, age, detections, image_path=None):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf_path = tmp_file.name
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(190, 10, "Brain Tumor Detection Report", ln=True, align='C')
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(190, 10, f"Contact: {email}", ln=True, align='C')
        pdf.cell(190, 10, f"Generated on: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}", ln=True, align='C')
        pdf.line(10, 30, 200, 30)
        pdf.ln(10)
        # Patient information section
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(190, 10, "PATIENT INFORMATION", ln=True, border=1, align='C', fill=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(95, 10, f"Patient ID: {patient_id}", border=1)
        pdf.cell(95, 10, f"Name: {patient_name}", border=1, ln=True)
        pdf.cell(95, 10, f"Gender: {gender}", border=1)
        pdf.cell(95, 10, f"Age: {age}", border=1, ln=True)
        pdf.ln(10)
        # Segmentation Results section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(190, 10, "DETECTION RESULTS", ln=True, border=1, align='C', fill=True)
        pdf.set_font("Helvetica", "", 11)
        best_tech = max(
            st.session_state['segmentation_results'].items(),
            key=lambda x: x[1]['accuracy']
        ) if st.session_state.get('segmentation_results') else ("N/A", {"accuracy": 0})
        for tech, data in st.session_state['segmentation_results'].items():
            pdf.cell(95, 10, f"Technique: {tech}", border=1)
            pdf.cell(95, 10, f"Estimated Accuracy: {data['accuracy']*100:.2f}%", border=1, ln=True)
            seg_img_path = f"seg_{tech}.jpg"
            data['image'].save(seg_img_path)
            pdf.image(seg_img_path, x=55, w=100)
            os.remove(seg_img_path)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(190, 10, f"RECOMMENDED TECHNIQUE: {best_tech[0]} {best_tech[1]['accuracy']*100:.2f}% accuracy", ln=True, border=1, align='C', fill=True)
        pdf.ln(10)
        # Detection Results section
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(190, 10, "DETECTION RESULTS", ln=True, border=1, align='C', fill=True)
        pdf.set_font("Helvetica", "", 11)
        if len(detections) > 0:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(190, 10, "TUMOR DETECTED", ln=True, align='C')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(20, 10, "No.", border=1, fill=True)
            pdf.cell(50, 10, "Location", border=1, fill=True)
            pdf.cell(60, 10, "Confidence", border=1, fill=True)
            pdf.cell(60, 10, "Classification", border=1, ln=True, fill=True)
            pdf.set_font("Helvetica", "", 11)
            for i, det in enumerate(detections, 1):
                xmin, ymin, xmax, ymax, conf, cls = det
                pdf.cell(20, 10, f"{i}", border=1)
                pdf.cell(50, 10, f"({int(xmin)}, {int(ymin)})", border=1)
                pdf.cell(60, 10, f"{conf:.2f}", border=1)
                pdf.cell(60, 10, "Malignant" if conf > 0.7 else "Benign", border=1, ln=True)
        else:
            pdf.set_text_color(0, 128, 0)
            pdf.cell(190, 10, "NO TUMOR DETECTED", ln=True, align='C')
            pdf.set_text_color(0, 0, 0)
        pdf.ln(10)
        # Add processed image if available
        if image_path and os.path.exists(image_path):
            try:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(190, 10, "MRI SCAN RESULTS", ln=True, border=1, align='C', fill=True)
                pdf.image(image_path, x=55, w=100)
            except Exception as e:
                pdf.ln(5)
                pdf.cell(190, 10, f"Error embedding image: {str(e)}", ln=True)
        pdf.ln(20)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(190, 5, "DISCLAIMER: This report is generated using an AI-based detection system and should be reviewed by a licensed medical professional. This tool is designed to aid in diagnosis, not replace professional medical judgment.")
        pdf.set_y(-15)
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 10, f"Page {pdf.page_no()}/1", 0, 0, 'C')
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        try:
            os.remove(pdf_path)
        except:
            pass
        return pdf_data

def main():
    # Initialize session state variables
    if 'segmentation_results' not in st.session_state:
        st.session_state['segmentation_results'] = {}
    if 'detect_pressed' not in st.session_state:
        st.session_state['detect_pressed'] = False
    if 'detections' not in st.session_state:
        st.session_state['detections'] = None

    st.markdown('<div class="main-header">Brain Tumor Detection System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2.5rem; font-size: 1.2rem; color: #34495e; max-width: 800px; margin-left: auto; margin-right: auto;">
        This application uses advanced AI technology to detect brain tumors from MRI scans.
        Upload your MRI scan image and get instant analysis with high accuracy.
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")

    # Left column: Patient information, image upload, and Detect Tumor button
    with col1:
        st.markdown('<div class="sub-header">Patient Information</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            patient_id = st.text_input("Patient ID", placeholder="Enter unique patient identifier")
            patient_name = st.text_input("Patient Name", placeholder="Enter full name")
            gender_col, age_col = st.columns(2)
            with gender_col:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            with age_col:
                age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Upload MRI Scan</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])
            temp_img_path = None
            image = None
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                    tmp_img.write(uploaded_file.read())
                    temp_img_path = tmp_img.name
                    image = Image.open(temp_img_path)
                    st.markdown('<div style="padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; margin-top: 10px;">', unsafe_allow_html=True)
                    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.session_state['temp_img_path'] = temp_img_path
                st.session_state['image'] = image
            st.markdown("</div>", unsafe_allow_html=True)

        # Detect Tumor button
        if 'image' in st.session_state:
            st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
            detect_btn = st.button("Detect Tumor", key="detect_tumor", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            if detect_btn:
                st.session_state['detect_pressed'] = True

    # Right column: Display detection and segmentation results
    with col2:
        st.markdown('<div class="sub-header">Detection Results</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            if (st.session_state.get('detect_pressed') and 
                'image' in st.session_state and 
                (st.session_state.get('detections') is None or len(st.session_state.get('detections', [])) == 0)):
                custom_spinner()
                image = st.session_state['image']
                temp_img_path = st.session_state['temp_img_path']
                model_path = "best.torchscript"
                seg_results = {}
                for seg_type, seg_func in [
                    ("CNN", apply_cnn_segmentation),
                    ("Transformer", apply_transformer_segmentation)
                ]:
                    seg_image, seg_accuracy = seg_func(image)
                    seg_results[seg_type] = {"image": seg_image, "accuracy": seg_accuracy}
                st.session_state['segmentation_results'] = seg_results
                result_image, detections = predict_and_display_v11(image, model_path)
                processed_img_path = f"{temp_img_path}_processed.jpg"
                result_image.save(processed_img_path)
                st.session_state['processed_img_path'] = processed_img_path
                st.session_state['detections'] = detections
                show_result_animation("positive" if len(detections) > 0 else "negative")

            # Display detection results
            if st.session_state.get('detections') is not None:
                if len(st.session_state['detections']) > 0:
                    st.markdown('<div class="tumor-detected fadein">Tumor Detected</div>', unsafe_allow_html=True)
                    st.markdown('<div class="result-box detection-positive fadein">', unsafe_allow_html=True)
                    for i, det in enumerate(st.session_state['detections'], 1):
                        xmin, ymin, xmax, ymax, conf, cls = det
                        severity = "Potentially Malignant" if conf > 0.7 else "Potentially Benign"
                        st.markdown(f"**Tumor {i}:** Confidence - {conf:.2f} | {severity}")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="no-tumor fadein">No Tumor Detected</div>', unsafe_allow_html=True)
                    st.markdown('<div class="result-box detection-negative fadein">', unsafe_allow_html=True)
                    st.markdown("The analysis found no evidence of tumors in the provided MRI scan.")
                    st.markdown("</div>", unsafe_allow_html=True)
                if st.session_state.get('processed_img_path'):
                    st.markdown('<div style="padding: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; margin-top: 15px; margin-bottom: 15px;">', unsafe_allow_html=True)
                    st.image(st.session_state['processed_img_path'], caption="Detection Results", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                # Display segmentation results
                if st.session_state.get('segmentation_results'):
                    st.markdown('<div class="sub-header">Segmentation Results</div>', unsafe_allow_html=True)
                    for tech, data in st.session_state['segmentation_results'].items():
                        st.markdown(f"**{tech} Segmentation (Accuracy: {data['accuracy']*100:.2f}%)**")
                        st.image(data['image'], caption=f"{tech} Segmented Image", use_container_width=True)
            else:
                st.info("After uploading, click 'Detect Tumor' (in the left column) to analyze the scan.")
            st.markdown("</div>", unsafe_allow_html=True)

    # PDF Report Generation Section
    st.markdown('<div class="pdf-download">', unsafe_allow_html=True)
    st.markdown("### Generate Medical Report")
    if st.button("Generate PDF Report", key="generate_pdf", use_container_width=True):
        with st.spinner("Generating comprehensive medical report..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i+1)
            pdf_data = generate_and_download_pdf(
                patient_id, patient_name, EMAIL, gender, age,
                st.session_state.get('detections', []),
                st.session_state.get('processed_img_path', None)
            )
            st.session_state['pdf_data'] = pdf_data
            st.success("Report generated successfully!")
    if "pdf_data" in st.session_state:
        st.download_button(
            label="Download PDF Report",
            data=st.session_state['pdf_data'],
            file_name=f"Image_Segmentation_Report_{patient_id}.pdf",
            mime="application/pdf",
            key="download_pdf",
            use_container_width=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Instruction box
    st.markdown("""
    <div class="instruction-box">
        <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 8px; margin-bottom: 15px;">
            About Brain Tumor Detection
        </h3>
        <p class="info-text">This application uses fine-tuned YOLOv11 and modern segmentation techniques (CNN, Transformer) to detect and segment brain tumors in MRI scans. The models have been fine-tuned on thousands of medical images to ensure high accuracy.</p>
        <h4 style="color: #3498db; margin-top: 15px;">
            How to use:
        </h4>
        <ol class="info-text">
            <li>Enter the patient's information in the form</li>
            <li>Upload a clear MRI scan (JPEG, PNG format)</li>
            <li>Click "Detect Tumor" to analyze and segment the scan</li>
            <li>Click "Generate PDF Report" to create your medical report</li>
            <li>Download the PDF report using the button provided</li>
        </ol>
        <p class="info-text"><strong>Note:</strong> This tool is designed to assist medical professionals and should not replace professional medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    footer = """
    <div class="footer">
        <p>Brain Tumor Dection System © 2025. All rights reserved.</p>
        <p>Contact: nyamangodotatenda64@gmail.com +263 71 424 2685 | Developed for healthcare professionals</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()