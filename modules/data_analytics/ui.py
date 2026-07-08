"""
Data Analytics UI Module
Streamlit interface for comprehensive data preprocessing and visualization
"""
import streamlit as st
import pandas as pd
from .preprocessing import DataPreprocessor
from .dashboard import DataDashboard


def render_data_analytics_page():
    """Render the enhanced Data Analytics page with comprehensive features"""
    
    st.title("📊 Data Analytics & Preprocessing")
    st.markdown("### Transform your data with powerful preprocessing and stunning visualizations!")
    st.markdown("---")
    
    # Custom CSS for colorful buttons
    st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    </style>
    """, unsafe_allow_html=True)
    
    # File Upload Section
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your dataset for preprocessing and analysis"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns!")
            
            # Initialize preprocessor and dashboard
            preprocessor = DataPreprocessor(df)
            dashboard = DataDashboard(df)
            
            # Create tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Data Overview", 
                "🔧 Preprocessing", 
                "📈 Advanced Visualizations",
                "💾 Processed Data"
            ])
            
            # ==================== TAB 1: DATA OVERVIEW ====================
            with tab1:
                st.header("📊 Data Overview")
                
                # Metrics
                dashboard.create_overview_metrics()
                st.markdown("---")
                
                # Data Quality Score
                dashboard.plot_data_quality_score()
                st.markdown("---")
                
                # Show first few rows
                st.subheader("🔍 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown("---")
                
                # Data types
                st.subheader("📋 Column Information")
                dashboard.show_data_types()
                st.markdown("---")
                
                # Statistical summary
                dashboard.show_statistical_summary()
                st.markdown("---")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("❌ Missing Values")
                    dashboard.plot_missing_values()
                
                with col2:
                    st.subheader("📊 Categorical Distribution")
                    dashboard.plot_categorical_distributions()
                
                st.markdown("---")
                
                # Pie charts
                dashboard.plot_categorical_pie_charts()
                st.markdown("---")
                
                # Numeric distributions
                st.subheader("📈 Numeric Distributions")
                dashboard.plot_numeric_distributions()
                
                st.markdown("---")
                
                # Correlation heatmap
                dashboard.plot_correlation_heatmap()
            
            # ==================== TAB 2: PREPROCESSING ====================
            with tab2:
                st.header("🔧 Data Preprocessing Toolkit")
                st.markdown("### Apply powerful transformations to your data")
                
                # Get column types
                numeric_cols, categorical_cols = preprocessor.get_column_types()
                
                # ===== SECTION 1: Missing Values =====
                st.markdown("---")
                st.subheader("1️⃣ Handle Missing Values")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    missing_strategy = st.selectbox(
                        "Select strategy for missing values",
                        ["None", "Mean (Numeric)", "Median (Numeric)", "Mode (Categorical)", 
                         "Forward Fill", "Backward Fill", "Fill with 0", "Drop Rows"],
                        key="missing_strategy"
                    )
                
                with col2:
                    if missing_strategy != "None":
                        if st.button("🚀 Apply Missing Value Strategy", key="apply_missing", use_container_width=True):
                            strategy_map = {
                                "Mean (Numeric)": "mean",
                                "Median (Numeric)": "median",
                                "Mode (Categorical)": "mode",
                                "Forward Fill": "ffill",
                                "Backward Fill": "bfill",
                                "Fill with 0": "constant",
                                "Drop Rows": "drop"
                            }
                            preprocessor.handle_missing_values(strategy_map[missing_strategy])
                            st.success(f"✅ Applied {missing_strategy} strategy!")
                
                # ===== SECTION 2: Encoding Methods =====
                st.markdown("---")
                st.subheader("2️⃣ Encoding Methods (Categorical → Numeric)")
                
                encoding_tabs = st.tabs(["🏷️ Label", "🔢 One-Hot", "📊 Ordinal", "📈 Frequency", "🎯 Target"])
                
                # Label Encoding
                with encoding_tabs[0]:
                    if len(categorical_cols) > 0:
                        label_encode_cols = st.multiselect(
                            "Select columns for Label Encoding",
                            categorical_cols,
                            key="label_encode",
                            help="Converts categories to numbers (0, 1, 2, ...)"
                        )
                        if st.button("🚀 Apply Label Encoding", key="apply_label", use_container_width=True):
                            if label_encode_cols:
                                preprocessor.apply_label_encoding(label_encode_cols)
                                st.success(f"✅ Applied Label Encoding to {len(label_encode_cols)} columns!")
                    else:
                        st.info("No categorical columns available")
                
                # One-Hot Encoding
                with encoding_tabs[1]:
                    if len(categorical_cols) > 0:
                        onehot_encode_cols = st.multiselect(
                            "Select columns for One-Hot Encoding",
                            categorical_cols,
                            key="onehot_encode",
                            help="Creates binary columns for each category"
                        )
                        if st.button("🚀 Apply One-Hot Encoding", key="apply_onehot", use_container_width=True):
                            if onehot_encode_cols:
                                preprocessor.apply_onehot_encoding(onehot_encode_cols)
                                st.success(f"✅ Applied One-Hot Encoding to {len(onehot_encode_cols)} columns!")
                    else:
                        st.info("No categorical columns available")
                
                # Ordinal Encoding
                with encoding_tabs[2]:
                    if len(categorical_cols) > 0:
                        ordinal_encode_cols = st.multiselect(
                            "Select columns for Ordinal Encoding",
                            categorical_cols,
                            key="ordinal_encode",
                            help="Converts ordered categories to numbers"
                        )
                        if st.button("🚀 Apply Ordinal Encoding", key="apply_ordinal", use_container_width=True):
                            if ordinal_encode_cols:
                                preprocessor.apply_ordinal_encoding(ordinal_encode_cols)
                                st.success(f"✅ Applied Ordinal Encoding to {len(ordinal_encode_cols)} columns!")
                    else:
                        st.info("No categorical columns available")
                
                # Frequency Encoding
                with encoding_tabs[3]:
                    if len(categorical_cols) > 0:
                        freq_encode_cols = st.multiselect(
                            "Select columns for Frequency Encoding",
                            categorical_cols,
                            key="freq_encode",
                            help="Replaces categories with their frequency"
                        )
                        if st.button("🚀 Apply Frequency Encoding", key="apply_freq", use_container_width=True):
                            if freq_encode_cols:
                                preprocessor.apply_frequency_encoding(freq_encode_cols)
                                st.success(f"✅ Applied Frequency Encoding to {len(freq_encode_cols)} columns!")
                    else:
                        st.info("No categorical columns available")
                
                # Target Encoding
                with encoding_tabs[4]:
                    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            target_encode_cols = st.multiselect(
                                "Select columns for Target Encoding",
                                categorical_cols,
                                key="target_encode",
                                help="Encodes based on target variable mean"
                            )
                        with col2:
                            target_col = st.selectbox(
                                "Select target column",
                                numeric_cols,
                                key="target_col"
                            )
                        
                        if st.button("🚀 Apply Target Encoding", key="apply_target", use_container_width=True):
                            if target_encode_cols:
                                preprocessor.apply_target_encoding(target_encode_cols, target_col)
                                st.success(f"✅ Applied Target Encoding to {len(target_encode_cols)} columns!")
                    else:
                        st.info("Need both categorical and numeric columns")
                
                # ===== SECTION 3: Scaling Methods =====
                st.markdown("---")
                st.subheader("3️⃣ Scaling & Normalization Methods")
                
                scaling_tabs = st.tabs(["📏 Standard", "📐 MinMax", "🛡️ Robust", "🔄 Normalize", "📊 Transform"])
                
                # Standard Scaling
                with scaling_tabs[0]:
                    if len(numeric_cols) > 0:
                        standard_scale_cols = st.multiselect(
                            "Select columns for Standard Scaling (Z-score)",
                            numeric_cols,
                            key="standard_scale",
                            help="Mean=0, Std=1"
                        )
                        if st.button("🚀 Apply Standard Scaling", key="apply_standard", use_container_width=True):
                            if standard_scale_cols:
                                preprocessor.apply_standard_scaling(standard_scale_cols)
                                st.success(f"✅ Applied Standard Scaling!")
                    else:
                        st.info("No numeric columns")
                
                # MinMax Scaling
                with scaling_tabs[1]:
                    if len(numeric_cols) > 0:
                        minmax_scale_cols = st.multiselect(
                            "Select columns for MinMax Scaling",
                            numeric_cols,
                            key="minmax_scale",
                            help="Scales to range [0, 1]"
                        )
                        if st.button("🚀 Apply MinMax Scaling", key="apply_minmax", use_container_width=True):
                            if minmax_scale_cols:
                                preprocessor.apply_minmax_scaling(minmax_scale_cols)
                                st.success(f"✅ Applied MinMax Scaling!")
                    else:
                        st.info("No numeric columns")
                
                # Robust Scaling
                with scaling_tabs[2]:
                    if len(numeric_cols) > 0:
                        robust_scale_cols = st.multiselect(
                            "Select columns for Robust Scaling",
                            numeric_cols,
                            key="robust_scale",
                            help="Uses median and IQR, robust to outliers"
                        )
                        if st.button("🚀 Apply Robust Scaling", key="apply_robust", use_container_width=True):
                            if robust_scale_cols:
                                preprocessor.apply_robust_scaling(robust_scale_cols)
                                st.success(f"✅ Applied Robust Scaling!")
                    else:
                        st.info("No numeric columns")
                
                # Normalization
                with scaling_tabs[3]:
                    if len(numeric_cols) > 0:
                        normalize_cols = st.multiselect(
                            "Select columns for L2 Normalization",
                            numeric_cols,
                            key="normalize",
                            help="Scales to unit norm"
                        )
                        if st.button("🚀 Apply Normalization", key="apply_normalize", use_container_width=True):
                            if normalize_cols:
                                preprocessor.apply_normalization(normalize_cols)
                                st.success(f"✅ Applied Normalization!")
                    else:
                        st.info("No numeric columns")
                
                # Transformations
                with scaling_tabs[4]:
                    if len(numeric_cols) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Log Transformation**")
                            log_cols = st.multiselect(
                                "Select columns",
                                numeric_cols,
                                key="log_transform",
                                help="Applies log(1+x)"
                            )
                            if st.button("🚀 Apply Log Transform", key="apply_log", use_container_width=True):
                                if log_cols:
                                    preprocessor.apply_log_transformation(log_cols)
                                    st.success(f"✅ Applied Log Transform!")
                        
                        with col2:
                            st.markdown("**Square Root Transformation**")
                            sqrt_cols = st.multiselect(
                                "Select columns",
                                numeric_cols,
                                key="sqrt_transform",
                                help="Applies sqrt(x)"
                            )
                            if st.button("🚀 Apply Sqrt Transform", key="apply_sqrt", use_container_width=True):
                                if sqrt_cols:
                                    preprocessor.apply_sqrt_transformation(sqrt_cols)
                                    st.success(f"✅ Applied Sqrt Transform!")
                    else:
                        st.info("No numeric columns")
                
                # ===== SECTION 4: Text Vectorization =====
                st.markdown("---")
                st.subheader("4️⃣ Text Vectorization")
                
                text_tabs = st.tabs(["📝 TF-IDF", "🔢 Count Vectorizer"])
                
                # TF-IDF
                with text_tabs[0]:
                    if len(categorical_cols) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            tfidf_col = st.selectbox(
                                "Select text column for TF-IDF",
                                ["None"] + categorical_cols,
                                key="tfidf_col"
                            )
                        
                        with col2:
                            max_features = st.number_input(
                                "Max features",
                                min_value=10,
                                max_value=500,
                                value=100,
                                key="tfidf_features"
                            )
                        
                        if tfidf_col != "None":
                            if st.button("🚀 Apply TF-IDF", key="apply_tfidf", use_container_width=True):
                                preprocessor.apply_tfidf(tfidf_col, max_features)
                                st.success(f"✅ Applied TF-IDF to {tfidf_col}!")
                    else:
                        st.info("No text columns available")
                
                # Count Vectorizer
                with text_tabs[1]:
                    if len(categorical_cols) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            count_col = st.selectbox(
                                "Select text column for Count Vectorization",
                                ["None"] + categorical_cols,
                                key="count_col"
                            )
                        
                        with col2:
                            count_features = st.number_input(
                                "Max features",
                                min_value=10,
                                max_value=500,
                                value=100,
                                key="count_features"
                            )
                        
                        if count_col != "None":
                            if st.button("🚀 Apply Count Vectorization", key="apply_count", use_container_width=True):
                                preprocessor.apply_count_vectorization(count_col, count_features)
                                st.success(f"✅ Applied Count Vectorization to {count_col}!")
                    else:
                        st.info("No text columns available")
                
                # ===== SECTION 5: Outlier Handling =====
                st.markdown("---")
                st.subheader("5️⃣ Outlier Handling")
                
                outlier_tabs = st.tabs(["🗑️ Remove", "📌 Cap"])
                
                # Remove Outliers
                with outlier_tabs[0]:
                    if len(numeric_cols) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            remove_outlier_cols = st.multiselect(
                                "Select columns to remove outliers",
                                numeric_cols,
                                key="remove_outliers",
                                help="Removes rows with outliers using IQR method"
                            )
                        
                        with col2:
                            iqr_multiplier = st.slider(
                                "IQR Multiplier",
                                min_value=1.0,
                                max_value=3.0,
                                value=1.5,
                                step=0.1,
                                key="iqr_mult_remove"
                            )
                        
                        if st.button("🚀 Remove Outliers", key="apply_remove_outliers", use_container_width=True):
                            if remove_outlier_cols:
                                preprocessor.remove_outliers_iqr(remove_outlier_cols, iqr_multiplier)
                                st.success(f"✅ Removed outliers from {len(remove_outlier_cols)} columns!")
                    else:
                        st.info("No numeric columns")
                
                # Cap Outliers
                with outlier_tabs[1]:
                    if len(numeric_cols) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            cap_outlier_cols = st.multiselect(
                                "Select columns to cap outliers",
                                numeric_cols,
                                key="cap_outliers",
                                help="Caps outliers to boundary values (winsorization)"
                            )
                        
                        with col2:
                            cap_multiplier = st.slider(
                                "IQR Multiplier",
                                min_value=1.0,
                                max_value=3.0,
                                value=1.5,
                                step=0.1,
                                key="iqr_mult_cap"
                            )
                        
                        if st.button("🚀 Cap Outliers", key="apply_cap_outliers", use_container_width=True):
                            if cap_outlier_cols:
                                preprocessor.cap_outliers_iqr(cap_outlier_cols, cap_multiplier)
                                st.success(f"✅ Capped outliers in {len(cap_outlier_cols)} columns!")
                    else:
                        st.info("No numeric columns")
                
                # ===== SECTION 6: Feature Engineering =====
                st.markdown("---")
                st.subheader("6️⃣ Feature Engineering")
                
                feature_tabs = st.tabs(["📊 Binning", "🔢 Polynomial", "✖️ Interactions"])
                
                # Binning
                with feature_tabs[0]:
                    if len(numeric_cols) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            bin_col = st.selectbox(
                                "Select column for binning",
                                ["None"] + numeric_cols,
                                key="bin_col"
                            )
                        
                        with col2:
                            num_bins = st.number_input(
                                "Number of bins",
                                min_value=2,
                                max_value=10,
                                value=5,
                                key="num_bins"
                            )
                        
                        if bin_col != "None":
                            if st.button("🚀 Create Bins", key="apply_binning", use_container_width=True):
                                preprocessor.create_binning(bin_col, num_bins)
                                st.success(f"✅ Created {num_bins} bins for {bin_col}!")
                    else:
                        st.info("No numeric columns")
                
                # Polynomial Features
                with feature_tabs[1]:
                    if len(numeric_cols) > 0:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            poly_cols = st.multiselect(
                                "Select columns for polynomial features",
                                numeric_cols,
                                key="poly_cols",
                                help="Creates polynomial and interaction features"
                            )
                        
                        with col2:
                            poly_degree = st.number_input(
                                "Degree",
                                min_value=2,
                                max_value=3,
                                value=2,
                                key="poly_degree"
                            )
                        
                        if st.button("🚀 Create Polynomial Features", key="apply_poly", use_container_width=True):
                            if poly_cols:
                                preprocessor.create_polynomial_features(poly_cols, poly_degree)
                                st.success(f"✅ Created polynomial features!")
                    else:
                        st.info("No numeric columns")
                
                # Interaction Features
                with feature_tabs[2]:
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            interact_col1 = st.selectbox(
                                "Select first column",
                                numeric_cols,
                                key="interact_col1"
                            )
                        
                        with col2:
                            interact_col2 = st.selectbox(
                                "Select second column",
                                numeric_cols,
                                key="interact_col2"
                            )
                        
                        if st.button("🚀 Create Interaction", key="apply_interact", use_container_width=True):
                            preprocessor.create_interaction_features(interact_col1, interact_col2)
                            st.success(f"✅ Created interaction: {interact_col1} × {interact_col2}!")
                    else:
                        st.info("Need at least 2 numeric columns")
                
                # ===== SECTION 7: Data Cleaning =====
                st.markdown("---")
                st.subheader("7️⃣ Data Cleaning")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🗑️ Remove Duplicates", key="remove_dups", use_container_width=True):
                        removed = preprocessor.remove_duplicates()
                        st.success(f"✅ Removed {removed} duplicate rows!")
                
                with col2:
                    if st.button("🔄 Reset Index", key="reset_idx", use_container_width=True):
                        preprocessor.reset_index()
                        st.success(f"✅ Reset index!")
            
            # ==================== TAB 3: ADVANCED VISUALIZATIONS ====================
            with tab3:
                st.header("📈 Advanced Visualizations")
                st.markdown("### Explore your data with comprehensive charts")
                
                # Box Plots
                dashboard.plot_box_plots()
                st.markdown("---")
                
                # Violin Plots
                dashboard.plot_violin_plots()
                st.markdown("---")
                
                # KDE Plots
                dashboard.plot_kde_plots()
                st.markdown("---")
                
                # Scatter Matrix
                dashboard.plot_scatter_matrix()
                st.markdown("---")
                
                # Outlier Analysis
                dashboard.plot_outlier_analysis()
            
            # ==================== TAB 4: PROCESSED DATA ====================
            with tab4:
                st.header("💾 Processed Data")
                
                processed_df = preprocessor.processed_df
                
                # Show metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Columns", len(df.columns))
                with col2:
                    st.metric("Processed Columns", len(processed_df.columns))
                with col3:
                    new_cols = len(processed_df.columns) - len(df.columns)
                    st.metric("New Features", new_cols, delta=f"+{new_cols}")
                with col4:
                    st.metric("Total Rows", len(processed_df))
                
                st.markdown("---")
                
                # Show processed data
                st.subheader("🔍 Processed Data Preview")
                st.dataframe(processed_df.head(20), use_container_width=True)
                
                st.markdown("---")
                
                # Download button
                csv = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Processed Data (CSV)",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    
    else:
        # Show placeholder
        st.info("👆 Please upload a CSV file to get started!")
        
        # Show example
        st.markdown("---")
        st.subheader("✨ Comprehensive Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔢 Encoding Methods:**
            - Label Encoding
            - One-Hot Encoding
            - Ordinal Encoding
            - Frequency Encoding
            - Target Encoding
            
            **📏 Scaling Methods:**
            - Standard Scaling
            - MinMax Scaling
            - Robust Scaling
            - L2 Normalization
            - Log Transform
            - Sqrt Transform
            """)
        
        with col2:
            st.markdown("""
            **📝 Text Processing:**
            - TF-IDF Vectorization
            - Count Vectorization
            
            **🎯 Outlier Handling:**
            - Remove Outliers (IQR)
            - Cap Outliers (Winsorization)
            
            **🔧 Feature Engineering:**
            - Binning
            - Polynomial Features
            - Interaction Features
            """)
        
        with col3:
            st.markdown("""
            **📊 Visualizations:**
            - Histograms & KDE
            - Box Plots
            - Violin Plots
            - Scatter Matrix
            - Correlation Heatmap
            - Pie Charts
            - Outlier Analysis
            - Data Quality Score
            
            **🧹 Data Cleaning:**
            - Missing Value Handling
            - Duplicate Removal
            """)
