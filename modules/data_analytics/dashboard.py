"""
Dashboard Module for Data Visualization
Creates interactive visualizations and statistics with comprehensive charts
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


class DataDashboard:
    """Interactive dashboard for data analytics with comprehensive visualizations"""
    
    def __init__(self, df):
        self.df = df
        # Set color palette
        self.colors = px.colors.qualitative.Set3
        self.sequential_colors = px.colors.sequential.Viridis
    
    def create_overview_metrics(self):
        """Create overview metrics cards"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📊 Total Rows",
                value=f"{len(self.df):,}"
            )
        
        with col2:
            st.metric(
                label="📋 Total Columns",
                value=len(self.df.columns)
            )
        
        with col3:
            missing = self.df.isnull().sum().sum()
            missing_pct = (missing / (len(self.df) * len(self.df.columns))) * 100
            st.metric(
                label="❌ Missing Values",
                value=f"{missing:,}",
                delta=f"{missing_pct:.1f}%"
            )
        
        with col4:
            memory = self.df.memory_usage(deep=True).sum() / 1024**2
            st.metric(
                label="💾 Memory Usage",
                value=f"{memory:.2f} MB"
            )
        
        with col5:
            duplicates = self.df.duplicated().sum()
            st.metric(
                label="🔄 Duplicates",
                value=f"{duplicates:,}"
            )
    
    def plot_missing_values(self):
        """Visualize missing values"""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) > 0:
            fig = px.bar(
                x=missing.values,
                y=missing.index,
                orientation='h',
                title='Missing Values by Column',
                labels={'x': 'Count', 'y': 'Column'},
                color=missing.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                template='plotly_dark',
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    
    def plot_numeric_distributions(self):
        """Plot distributions of numeric columns with histograms"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            # Select columns for visualization
            cols_to_plot = numeric_cols[:min(6, len(numeric_cols))]
            
            rows = (len(cols_to_plot) + 2) // 3
            fig = make_subplots(
                rows=rows, cols=3,
                subplot_titles=[col for col in cols_to_plot]
            )
            
            for idx, col in enumerate(cols_to_plot):
                row = idx // 3 + 1
                col_pos = idx % 3 + 1
                
                fig.add_trace(
                    go.Histogram(
                        x=self.df[col],
                        name=col,
                        marker_color=self.colors[idx % len(self.colors)],
                        showlegend=False
                    ),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                template='plotly_dark',
                height=300 * rows,
                title_text="📊 Numeric Column Distributions"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_box_plots(self):
        """Create box plots for numeric columns to show outliers"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            st.subheader("📦 Box Plots (Outlier Detection)")
            
            cols_to_plot = numeric_cols[:min(6, len(numeric_cols))]
            
            fig = go.Figure()
            
            for idx, col in enumerate(cols_to_plot):
                fig.add_trace(go.Box(
                    y=self.df[col],
                    name=col,
                    marker_color=self.colors[idx % len(self.colors)],
                    boxmean='sd'
                ))
            
            fig.update_layout(
                template='plotly_dark',
                height=500,
                title='Box Plots for Numeric Columns',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_violin_plots(self):
        """Create violin plots for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            st.subheader("🎻 Violin Plots (Distribution Shape)")
            
            cols_to_plot = numeric_cols[:min(6, len(numeric_cols))]
            
            fig = go.Figure()
            
            for idx, col in enumerate(cols_to_plot):
                fig.add_trace(go.Violin(
                    y=self.df[col],
                    name=col,
                    marker_color=self.colors[idx % len(self.colors)],
                    box_visible=True,
                    meanline_visible=True
                ))
            
            fig.update_layout(
                template='plotly_dark',
                height=500,
                title='Violin Plots for Numeric Columns',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_kde_plots(self):
        """Create KDE (Kernel Density Estimation) plots"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            st.subheader("📈 KDE Plots (Density Estimation)")
            
            cols_to_plot = numeric_cols[:min(4, len(numeric_cols))]
            
            fig = go.Figure()
            
            for idx, col in enumerate(cols_to_plot):
                # Calculate KDE
                data = self.df[col].dropna()
                if len(data) > 0:
                    from scipy import stats
                    density = stats.gaussian_kde(data)
                    xs = np.linspace(data.min(), data.max(), 200)
                    
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=density(xs),
                        name=col,
                        fill='tozeroy',
                        line=dict(color=self.colors[idx % len(self.colors)])
                    ))
            
            fig.update_layout(
                template='plotly_dark',
                height=400,
                title='Kernel Density Estimation',
                xaxis_title='Value',
                yaxis_title='Density'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for numeric columns"""
        numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
        
        if len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                template='plotly_dark',
                title='🔥 Correlation Heatmap',
                height=600,
                xaxis={'side': 'bottom'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_scatter_matrix(self):
        """Create scatter matrix (pair plot) for numeric columns"""
        numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
        
        if len(numeric_df.columns) >= 2:
            st.subheader("🔷 Scatter Matrix (Pair Plot)")
            
            # Limit to 5 columns for readability
            cols_to_plot = numeric_df.columns[:min(5, len(numeric_df.columns))]
            
            fig = px.scatter_matrix(
                self.df,
                dimensions=cols_to_plot,
                title="Scatter Matrix",
                color_discrete_sequence=self.colors
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_categorical_distributions(self):
        """Plot distributions of categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            # Select first categorical column with reasonable unique values
            for col in categorical_cols:
                unique_count = self.df[col].nunique()
                if unique_count <= 20:  # Only plot if not too many categories
                    value_counts = self.df[col].value_counts().head(10)
                    
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f'Distribution of {col} (Top 10)',
                        labels={'x': col, 'y': 'Count'},
                        color=value_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        template='plotly_dark',
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    break
    
    def plot_categorical_pie_charts(self):
        """Create pie charts for categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            st.subheader("🥧 Categorical Distribution (Pie Charts)")
            
            # Find suitable columns
            suitable_cols = []
            for col in categorical_cols:
                unique_count = self.df[col].nunique()
                if 2 <= unique_count <= 10:
                    suitable_cols.append(col)
            
            if suitable_cols:
                cols_to_plot = suitable_cols[:min(3, len(suitable_cols))]
                
                cols = st.columns(len(cols_to_plot))
                
                for idx, col in enumerate(cols_to_plot):
                    with cols[idx]:
                        value_counts = self.df[col].value_counts().head(8)
                        
                        fig = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f'{col}',
                            color_discrete_sequence=self.colors
                        )
                        
                        fig.update_layout(
                            template='plotly_dark',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
    
    def plot_outlier_analysis(self):
        """Analyze and visualize outliers using IQR method"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numeric_cols) > 0:
            st.subheader("🎯 Outlier Analysis (IQR Method)")
            
            outlier_counts = {}
            
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_counts[col] = len(outliers)
            
            # Create bar chart
            outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=['Column', 'Outlier Count'])
            outlier_df = outlier_df[outlier_df['Outlier Count'] > 0].sort_values('Outlier Count', ascending=False)
            
            if len(outlier_df) > 0:
                fig = px.bar(
                    outlier_df,
                    x='Column',
                    y='Outlier Count',
                    title='Outlier Count by Column',
                    color='Outlier Count',
                    color_continuous_scale='Reds'
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No outliers detected!")
    
    def show_data_types(self):
        """Display data types information"""
        dtype_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes.values,
            'Non-Null Count': self.df.count().values,
            'Null Count': self.df.isnull().sum().values,
            'Unique Values': [self.df[col].nunique() for col in self.df.columns]
        })
        
        st.dataframe(dtype_df, use_container_width=True, height=400)
    
    def show_statistical_summary(self):
        """Show statistical summary"""
        st.subheader("📈 Statistical Summary")
        st.dataframe(self.df.describe(), use_container_width=True)
    
    def plot_data_quality_score(self):
        """Calculate and visualize data quality score"""
        st.subheader("⭐ Data Quality Score")
        
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        
        duplicate_score = ((len(self.df) - self.df.duplicated().sum()) / len(self.df)) * 100
        
        # Overall quality score
        quality_score = (completeness_score + duplicate_score) / 2
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Completeness", f"{completeness_score:.1f}%")
        
        with col2:
            st.metric("Uniqueness", f"{duplicate_score:.1f}%")
        
        with col3:
            st.metric("Overall Quality", f"{quality_score:.1f}%")
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=quality_score,
            title={'text': "Data Quality Score"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "gray"},
                    {'range': [75, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
