from ultralytics import YOLO
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="VehicleVision AI",
    page_icon="🚗",
    layout="wide"
)


@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


st.markdown("""
<style>

.main-header{
    background: linear-gradient(135deg,#2563eb,#7c3aed);
    padding:35px;
    border-radius:15px;
    text-align:center;
    margin-bottom:25px;
}

.footer{
    text-align:center;
    color:#9ca3af;
    padding:15px;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)


with st.sidebar:

    st.title("🚗 VehicleVision AI")

    st.markdown("---")

    st.subheader("📌 Project")

    st.info("""
Vehicle Detection using a custom-trained YOLOv8 model.

✓ Real-time Detection

✓ Bounding Boxes

✓ Confidence Scores

✓ Multiple Vehicle Classes
""")

    st.markdown("---")

    st.subheader("🧠 Model")

    st.write("YOLOv8 Nano")
    st.write("Custom Trained")
    st.write("7 Vehicle Classes")

    st.markdown("---")

    st.subheader("👨‍💻 Developer")

    st.write("Vansh Garg")
    st.write("B.Tech CSE")
    st.write("Machine Learning Enthusiast")


st.markdown("""
<div class="main-header">
<h1 style="color:white;">🚗 VehicleVision AI</h1>
<h4 style="color:white;">
Real-Time Vehicle Detection using YOLOv8
</h4>
<p style="color:#e5e7eb;">
Detect and classify vehicles from uploaded images
</p>
</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎯 mAP@50", "88.3%")

with col2:
    st.metric("📸 Dataset Images", "5,052")

with col3:
    st.metric("🚘 Classes", "7")


st.markdown("---")

st.subheader("📌 About The Project")

st.info("""
VehicleVision AI is an Object Detection application built using YOLOv8.

The model detects and classifies multiple vehicle types including buses, cars, ambulances, trucks and emergency vehicles.

Upload any road or traffic image and the model will automatically detect vehicles and display confidence scores.
""")



st.markdown("---")

st.subheader("🚘 Supported Vehicle Classes")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.success("🚍 Bus")
    st.success("🚗 Cars")

with c2:
    st.success("🚐 Jeepney")
    st.success("🚑 Ambulance")

with c3:
    st.success("🚒 Firetruck")
    st.success("🚓 Police")

with c4:
    st.success("🚚 Truck")



st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload a Vehicle Image",
    type=["jpg", "jpeg", "png"]
)



if uploaded_file:

    image = Image.open(uploaded_file)

    with st.spinner("Detecting vehicles..."):

        results = model.predict(
            image,
            conf=0.25
        )

    annotated_image = results[0].plot()

    vehicle_count = len(results[0].boxes)

    vehicle_types = {}

    for box in results[0].boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name in vehicle_types:
            vehicle_types[class_name] += 1
        else:
            vehicle_types[class_name] = 1

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🚘 Vehicles Detected",
            vehicle_count
        )

    with col2:
        st.metric(
            "📊 Vehicle Types",
            len(vehicle_types)
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("🎯 Detection Result")
        st.image(
            annotated_image,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("📋 Detection Summary")

    left, right = st.columns(2)

    with left:

        for vehicle, count in vehicle_types.items():
            st.info(f"{vehicle}: {count}")

    with right:

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            st.success(
                f"{class_name} ({confidence:.2%})"
            )


st.markdown("""
<div class="footer">
🚗 VehicleVision AI • Built with YOLOv8 • Streamlit • Ultralytics • © 2026 Vansh Garg
</div>
""", unsafe_allow_html=True)
