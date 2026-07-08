"""
Computer Vision UI Module
Enhanced Streamlit interface for YOLO training and inference with dataset upload
"""
import streamlit as st
from PIL import Image
import os
import tempfile
import zipfile
import shutil
from pathlib import Path
from .yolo_trainer import YOLOTrainer, create_data_yaml
from .detection import ObjectDetector
from .segmentation import InstanceSegmentation


def render_computer_vision_page():
    """Render the enhanced Computer Vision page with dataset upload"""
    
    st.title("🎯 Computer Vision with YOLO11n")
    st.markdown("### Train custom models and run inference with state-of-the-art YOLO11n")
    st.markdown("---")
    
    # Custom CSS for colorful buttons
    st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .success-box {
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Task Selection
    st.header("🔧 Select Task")
    task_type = st.selectbox(
        "Choose your computer vision task",
        ["Object Detection", "Instance Segmentation", "Oriented Bounding Box (OBB)", "Pose Estimation"],
        help="Select the type of computer vision task you want to perform"
    )
    
    task_map = {
        "Object Detection": "detection",
        "Instance Segmentation": "segmentation",
        "Oriented Bounding Box (OBB)": "obb",
        "Pose Estimation": "pose"
    }
    
    selected_task = task_map[task_type]
    
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎨 Inference", "🏋️ Training", "ℹ️ Model Info"])
    
    # ==================== TAB 1: INFERENCE ====================
    with tab1:
        st.header("🎨 Run Inference")
        st.markdown("### Upload an image and detect objects in real-time!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Upload image
            uploaded_image = st.file_uploader(
                "Upload an image for inference",
                type=['jpg', 'jpeg', 'png'],
                key="inference_image"
            )
            
            # Confidence threshold
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.05,
                key="conf_threshold",
                help="Lower values detect more objects but may include false positives"
            )
            
            # IOU threshold
            iou_threshold = st.slider(
                "IOU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                key="iou_threshold",
                help="Higher values reduce overlapping detections"
            )
        
        with col2:
            if uploaded_image is not None:
                # Display original image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if uploaded_image is not None:
            if st.button("🚀 Run Inference", key="run_inference", use_container_width=True):
                with st.spinner("🔍 Analyzing image..."):
                    try:
                        # Load appropriate model
                        if selected_task == "detection":
                            detector = ObjectDetector()
                            results = detector.detect(image, conf_threshold)
                            annotated = detector.draw_detections(image, results)
                            detections = detector.get_detection_info(results)
                            
                            # Display results
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("📊 Detection Results")
                                st.image(annotated, caption="Detected Objects", use_column_width=True)
                            
                            with col2:
                                st.subheader(f"📋 Detections ({len(detections)} found)")
                                if len(detections) > 0:
                                    for i, det in enumerate(detections):
                                        st.markdown(f"""
                                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                                    padding: 10px; border-radius: 8px; margin: 5px 0; color: white;'>
                                        <strong>Detection {i+1}:</strong><br>
                                        🏷️ Class: <code>{det['class']}</code><br>
                                        📊 Confidence: <code>{det['confidence']:.2%}</code>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No objects detected. Try lowering the confidence threshold.")
                        
                        elif selected_task == "segmentation":
                            segmenter = InstanceSegmentation()
                            results = segmenter.segment(image, conf_threshold)
                            annotated = segmenter.draw_segmentation(image, results)
                            segments = segmenter.get_segmentation_info(results)
                            
                            # Display results
                            st.markdown("---")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("📊 Segmentation Results")
                                st.image(annotated, caption="Segmented Objects", use_column_width=True)
                            
                            with col2:
                                st.subheader(f"📋 Segments ({len(segments)} found)")
                                if len(segments) > 0:
                                    for i, seg in enumerate(segments):
                                        st.markdown(f"""
                                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                                    padding: 10px; border-radius: 8px; margin: 5px 0; color: white;'>
                                        <strong>Segment {i+1}:</strong><br>
                                        🏷️ Class: <code>{seg['class']}</code><br>
                                        📊 Confidence: <code>{seg['confidence']:.2%}</code>
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No objects segmented. Try lowering the confidence threshold.")
                        
                        else:
                            st.info(f"Inference for {task_type} will be available after training!")
                        
                        st.success("✅ Inference completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during inference: {str(e)}")
                        st.exception(e)
    
    # ==================== TAB 2: TRAINING ====================
    with tab2:
        st.header("🏋️ Train Custom YOLO Model")
        st.markdown("### Upload your annotated dataset and train a custom model")
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; border-radius: 10px; color: white; margin: 10px 0;'>
        <strong>🎯 Current Task:</strong> {task_type}
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset Upload Section
        st.markdown("---")
        st.subheader("📦 Dataset Upload")
        
        upload_method = st.radio(
            "Choose upload method",
            ["Upload ZIP file", "Use existing path"],
            horizontal=True
        )
        
        dataset_path = None
        
        if upload_method == "Upload ZIP file":
            st.markdown("""
            **📁 Dataset Structure (inside ZIP):**
            ```
            dataset.zip
            ├── train/
            │   ├── images/
            │   │   ├── img1.jpg
            │   │   └── img2.jpg
            │   └── labels/
            │       ├── img1.txt
            │       └── img2.txt
            ├── valid/
            │   ├── images/
            │   └── labels/
            └── test/ (optional)
                ├── images/
                └── labels/
            ```
            
            **Label Format (YOLO):** `class_id x_center y_center width height` (normalized 0-1)
            """)
            
            uploaded_zip = st.file_uploader(
                "Upload dataset ZIP file",
                type=['zip'],
                key="dataset_zip",
                help="Upload a ZIP file containing your annotated dataset"
            )
            
            if uploaded_zip is not None:
                # Extract ZIP
                with st.spinner("📦 Extracting dataset..."):
                    try:
                        # Create temp directory
                        temp_dir = tempfile.mkdtemp()
                        zip_path = os.path.join(temp_dir, "dataset.zip")
                        
                        # Save uploaded file
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_zip.getbuffer())
                        
                        # Extract
                        extract_dir = os.path.join(temp_dir, "dataset")
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        dataset_path = extract_dir
                        
                        # Validate dataset structure
                        train_images = os.path.join(extract_dir, "train", "images")
                        train_labels = os.path.join(extract_dir, "train", "labels")
                        valid_images = os.path.join(extract_dir, "valid", "images")
                        valid_labels = os.path.join(extract_dir, "valid", "labels")
                        
                        if os.path.exists(train_images) and os.path.exists(train_labels):
                            train_img_count = len([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
                            train_lbl_count = len([f for f in os.listdir(train_labels) if f.endswith('.txt')])
                            
                            valid_img_count = 0
                            valid_lbl_count = 0
                            if os.path.exists(valid_images) and os.path.exists(valid_labels):
                                valid_img_count = len([f for f in os.listdir(valid_images) if f.endswith(('.jpg', '.jpeg', '.png'))])
                                valid_lbl_count = len([f for f in os.listdir(valid_labels) if f.endswith('.txt')])
                            
                            st.success("✅ Dataset extracted successfully!")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Train Images", train_img_count)
                            with col2:
                                st.metric("Train Labels", train_lbl_count)
                            with col3:
                                st.metric("Valid Images", valid_img_count)
                            with col4:
                                st.metric("Valid Labels", valid_lbl_count)
                            
                            # Show sample images
                            st.markdown("---")
                            st.subheader("🖼️ Sample Images")
                            sample_images = [f for f in os.listdir(train_images) if f.endswith(('.jpg', '.jpeg', '.png'))][:3]
                            cols = st.columns(len(sample_images))
                            for idx, img_name in enumerate(sample_images):
                                with cols[idx]:
                                    img_path = os.path.join(train_images, img_name)
                                    st.image(img_path, caption=img_name, use_column_width=True)
                        else:
                            st.error("❌ Invalid dataset structure! Please check the format.")
                            dataset_path = None
                        
                    except Exception as e:
                        st.error(f"❌ Error extracting dataset: {str(e)}")
                        dataset_path = None
        
        else:
            # Use existing path
            dataset_path = st.text_input(
                "Dataset root path",
                placeholder="path/to/dataset",
                key="dataset_path_input",
                help="Path to your dataset folder"
            )
            
            if dataset_path and os.path.exists(dataset_path):
                st.success(f"✅ Dataset path found: {dataset_path}")
        
        # Training Configuration
        st.markdown("---")
        st.subheader("⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Class names input
            class_names_input = st.text_area(
                "Enter class names (one per line)",
                placeholder="person\ncar\ndog\ncat",
                height=150,
                key="class_names",
                help="List all classes in your dataset"
            )
            
            epochs = st.slider(
                "Training Epochs",
                min_value=1,
                max_value=300,
                value=50,
                step=10,
                key="epochs",
                help="More epochs = better training but takes longer"
            )
        
        with col2:
            batch_size = st.selectbox(
                "Batch Size",
                [4, 8, 16, 32],
                index=2,
                key="batch_size",
                help="Higher batch size = faster training but needs more memory"
            )
            
            img_size = st.selectbox(
                "Image Size",
                [320, 416, 640, 1280],
                index=2,
                key="img_size",
                help="Higher resolution = better accuracy but slower training"
            )
            
            device = st.selectbox(
                "Device",
                ["cpu", "cuda"],
                index=0,
                key="device",
                help="Use CUDA for GPU acceleration if available"
            )
            
            patience = st.number_input(
                "Early Stopping Patience",
                min_value=0,
                max_value=50,
                value=10,
                key="patience",
                help="Stop training if no improvement for N epochs (0 = disabled)"
            )
        
        st.markdown("---")
        
        # Start training button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_training = st.button(
                "🚀 Start Training",
                key="start_training",
                use_container_width=True
            )
        
        if start_training:
            if not class_names_input or not dataset_path:
                st.error("❌ Please provide class names and upload/specify dataset path!")
            else:
                with st.spinner("🏋️ Training in progress... This may take a while."):
                    try:
                        # Parse class names
                        class_names = [name.strip() for name in class_names_input.split('\n') if name.strip()]
                        
                        st.info(f"📚 Training with {len(class_names)} classes: {', '.join(class_names)}")
                        
                        # Create data.yaml
                        train_path = os.path.join(dataset_path, "train", "images")
                        val_path = os.path.join(dataset_path, "valid", "images")
                        yaml_path = os.path.join(dataset_path, "data.yaml")
                        
                        create_data_yaml(train_path, val_path, class_names, yaml_path)
                        
                        # Initialize trainer
                        trainer = YOLOTrainer(task_type=selected_task)
                        trainer.load_model()
                        
                        st.info(f"📦 Model loaded: {trainer.model_map[selected_task]}")
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("🏋️ Training started...")
                        
                        # Train
                        results = trainer.train(
                            data_yaml=yaml_path,
                            epochs=epochs,
                            imgsz=img_size,
                            batch=batch_size,
                            device=device,
                            patience=patience
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Training completed!")
                        
                        # Show results
                        st.markdown("---")
                        st.subheader("📊 Training Results")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Task", task_type)
                        with col2:
                            st.metric("Epochs", epochs)
                        with col3:
                            st.metric("Classes", len(class_names))
                        with col4:
                            st.metric("Status", "✅ Completed")
                        
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                    padding: 20px; border-radius: 10px; color: white; margin: 20px 0;'>
                        <h3>🎉 Model Trained Successfully!</h3>
                        <p>Your custom YOLO11n model has been trained and saved.</p>
                        <p>📁 Check the <code>runs/{}/train</code> folder for:</p>
                        <ul>
                            <li>✅ Trained model weights</li>
                            <li>📊 Training metrics and graphs</li>
                            <li>📈 Confusion matrix</li>
                            <li>🖼️ Sample predictions</li>
                        </ul>
                        </div>
                        """.format(selected_task), unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Training error: {str(e)}")
                        st.exception(e)
                        st.info("💡 Make sure your dataset is properly formatted and the path is correct.")
    
    # ==================== TAB 3: MODEL INFO ====================
    with tab3:
        st.header("ℹ️ Model Information")
        st.markdown("### Learn about YOLO11n capabilities")
        
        try:
            trainer = YOLOTrainer(task_type=selected_task)
            info = trainer.get_model_info()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Task Type", info['task'].title())
            
            with col2:
                st.metric("Model", info['model_name'])
            
            with col3:
                params = info['parameters'] / 1e6
                st.metric("Parameters", f"{params:.2f}M")
            
            with col4:
                st.metric("Size", "Nano")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📚 About YOLO11n")
                st.markdown(f"""
                **YOLO11n** is the nano version of YOLO11, optimized for:
                - ⚡ **Fast inference** - Real-time performance
                - 💾 **Low memory** - Runs on edge devices
                - 🎯 **High accuracy** - State-of-the-art results
                - 🔧 **Easy training** - Transfer learning ready
                
                **Current Task:** {task_type}
                """)
            
            with col2:
                st.subheader("🎯 Model Capabilities")
                st.markdown("""
                **Supported Tasks:**
                - 🔍 Object Detection
                - 🎨 Instance Segmentation
                - 📐 Oriented Bounding Boxes
                - 🤸 Pose Estimation
                
                **Training Features:**
                - 📦 Transfer learning
                - 🔄 Data augmentation
                - ⚡ Mixed precision
                - 🛑 Early stopping
                """)
            
            st.markdown("---")
            
            st.subheader("💡 Tips for Best Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Dataset Preparation:**
                - ✅ Use high-quality images
                - ✅ Balance class distribution
                - ✅ Include diverse examples
                - ✅ Proper annotation format
                - ✅ Train/Valid split (80/20)
                """)
            
            with col2:
                st.markdown("""
                **Training Tips:**
                - 🎯 Start with 50-100 epochs
                - 📊 Monitor validation metrics
                - ⚙️ Adjust batch size for GPU
                - 🔄 Use data augmentation
                - 💾 Save best checkpoints
                """)
            
        except Exception as e:
            st.error(f"Error loading model info: {str(e)}")
