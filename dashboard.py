import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from data_simulator import DataSimulator
from london_map import create_london_sentiment_map
from uk_map import create_uk_sentiment_map
from streamlit_folium import folium_static
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from langchain.llms import HuggingFaceEndpoint
import numpy as np
from fpdf import FPDF
import os
import base64
import pydeck as pdk

# Initialize simulator
simulator = DataSimulator()

HUGGINGFACE_API_TOKEN = "HF_TOKEN"
if not HUGGINGFACE_API_TOKEN:
    st.error("Please set HUGGINGFACE_API_TOKEN in secrets or environment variables")
    st.stop()

@st.cache_data
def load_data():
    return simulator.generate_sample_data(500)

def check_for_alerts(df):
    """Check for critical patterns in the data."""
    alerts = []
    
    # Regional alerts
    regional_stats = df.groupby('region')['is_negative'].mean()
    if regional_stats.max() > 0.3:
        alerts.append({
            "type": "region",
            "message": f"High negative sentiment in {regional_stats.idxmax()} ({(regional_stats.max()*100):.1f}%)",
            "severity": "high"
        })
    
    # Product alerts
    product_stats = df.groupby('product')['is_negative'].mean()
    if product_stats.max() > 0.25:
        alerts.append({
            "type": "product",
            "message": f"Pension product issue detected with {product_stats.idxmax()} ({(product_stats.max()*100):.1f}%)",
            "severity": "medium"
        })
    
    # Channel alerts
    channel_stats = df.groupby('channel')['is_negative'].mean()
    if channel_stats.max() > 0.35:
        alerts.append({
            "type": "channel",
            "message": f"Channel issue detected with {channel_stats.idxmax()} ({(channel_stats.max()*100):.1f}%)",
            "severity": "medium"
        })
    
    return alerts

def detect_anomalies(df):
    """Identify unusual feedback patterns using Isolation Forest"""
    vectorizer = TfidfVectorizer(max_features=500)
    text_vectors = vectorizer.fit_transform(df['feedback_text'])
    
    model = IsolationForest(contamination=0.1, random_state=42)
    anomalies = model.fit_predict(text_vectors)
    
    df['is_anomaly'] = anomalies == -1
    return df[df['is_anomaly']].sort_values('score', ascending=False)

def generate_summary_pdf(text, filename="executive_summary.pdf"):
    # Create temp directory if it doesn't exist
    os.makedirs("C:\\temp", exist_ok=True)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    
    output_path = os.path.join("C:\\temp", filename)
    pdf.output(output_path)
    
    return output_path
    
def generate_executive_summary(df):
    """Generate insights summary using HuggingFace LLM"""
    sentiment_dist = df['sentiment'].value_counts(normalize=True).to_dict()
    top_issues = df[df['is_negative']].groupby('product')['feedback_text'].count().nlargest(3)
    
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        temperature=0.3,
        huggingfacehub_api_token=HUGGINGFACE_API_TOKEN
    )
    
    prompt = f"""
    [INST] As a banking CX analyst, create a concise executive summary with:
    1. Key sentiment metrics (Positive: {sentiment_dist.get('POSITIVE', 0):.1%}, Negative: {sentiment_dist.get('NEGATIVE', 0):.1%})
    2. Top 3 pension product issues: {', '.join(top_issues.index.tolist())}
    3. Three actionable recommendations for pension product improvement
    
    Use bullet points and professional banking language. [/INST]
    """
    return llm.invoke(prompt)

def main():
    st.set_page_config(
        page_title="Lloyds Pension Sentiment Dashboard",
        layout="wide",
        page_icon="💼",
        initial_sidebar_state="expanded"
    )
    
    # =============================================
    # Updated Lloyds Green Theme CSS
    # =============================================
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #f8f9fa;
            background-image: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Lloyds color scheme - Updated to official green */
        :root {
            --lloyds-primary: #00843D;  /* Official Lloyds green */
            --lloyds-secondary: #003D2B; /* Darker green */
            --lloyds-accent: #00A551;   /* Brighter green */
            --lloyds-light: #E6F4F1;    /* Light green tint */
            --lloyds-dark: #231F20;     /* For text */
            --lloyds-alert: #D4122F;    /* For alerts/errors */
        }
        
        /* Header styling */
        header .decoration {
            background-color: var(--lloyds-primary) !important;
            height: 5px;
        }
        
        /* Sidebar styling */
        .st-emotion-cache-6qob1r {
            background-color: var(--lloyds-primary);
            color: white;
            border-right: 1px solid #ddd;
        }
        .st-emotion-cache-6qob1r .st-emotion-cache-16txtl3 {
            color: white;
        }
        
        /* Metric cards - Lloyds style */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 132, 61, 0.1);
            border-left: 4px solid var(--lloyds-primary);
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0, 132, 61, 0.15);
        }
        .metric-card h2 {
            color: var(--lloyds-primary);
            font-size: 28px;
            margin-top: 10px;
        }
        
        /* Tabs styling */
        .st-emotion-cache-1qg05tj {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Buttons */
        .st-emotion-cache-7ym5gk {
            background-color: var(--lloyds-primary) !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
        }
        
        /* Alerts */
        .stAlert {
            border-radius: 8px;
        }
        
        /* Custom title styling */
        .dashboard-title {
            color: var(--lloyds-primary);
            font-weight: 700;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .dashboard-subtitle {
            color: var(--lloyds-dark);
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        
        /* Custom cards */
        .custom-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid #eee;
        }
        
        /* Map container */
        .map-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Anomaly detection highlight */
        .anomaly-highlight {
            background-color: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        /* Updated tab styling */
        .st-emotion-cache-1qg05tj button {
            color: var(--lloyds-dark) !important;
        }
        .st-emotion-cache-1qg05tj button[aria-selected="true"] {
            color: var(--lloyds-primary) !important;
            font-weight: 600 !important;
            border-bottom: 2px solid var(--lloyds-primary) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # =============================================
    # Premium Header Section
    # =============================================
    st.markdown("""
    <div style="background-color: #00843D; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40" fill="white" style="vertical-align: middle; margin-right: 15px;">
                        <path d="M20 0c11.046 0 20 8.954 20 20s-8.954 20-20 20S0 31.046 0 20 8.954 0 20 0zm0 3.75C10.833 3.75 3.75 10.833 3.75 20S10.833 36.25 20 36.25 36.25 29.167 36.25 20 29.167 3.75 20 3.75zm-1.25 7.5h2.5v15h-2.5v-15zm0-7.5h2.5v5h-2.5v-5z"/>
                    </svg>
                    Lloyds Banking Group
                </h1>
                <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 55px; font-size: 1.2rem;">
                    Pension Products Customer Sentiment Dashboard
                </p>
            </div>
            <div style="background-color: white; padding: 10px 15px; border-radius: 8px; display: flex; align-items: center;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#00843D">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm4.59-12.42L10 14.17l-2.59-2.58L6 13l4 4 8-8z"/>
                </svg>
                <span style="color: #00843D; font-weight: 600; margin-left: 8px;">LIVE DATA</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # =============================================
    # Data Loading and Filtering (unchanged)
    # =============================================
    df = load_data()
    
    # Sidebar filters with premium styling
    with st.sidebar:
        st.markdown("""
        <div style="color: white; font-size: 1.2rem; font-weight: 600; margin-bottom: 15px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="white" style="vertical-align: middle; margin-right: 8px;">
                <path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"/>
            </svg>
            Data Filters
        </div>
        """, unsafe_allow_html=True)
        
        selected_channels = st.multiselect(
            "Select Channels", 
            df['channel'].unique(), 
            default=df['channel'].unique()
        )
        
        selected_regions = st.multiselect(
            "Select Regions",
            df['region'].unique(),
            default=df['region'].unique()
        )
        
        selected_products = st.multiselect(
            "Select Pension Products",
            df['product'].unique(),
            default=df['product'].unique()
        )
        
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Filter data (unchanged)
    filtered_df = df[
        (df['channel'].isin(selected_channels)) & 
        (df['region'].isin(selected_regions)) &
        (df['product'].isin(selected_products))
    ]
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= date_range[0]) &
            (filtered_df['timestamp'].dt.date <= date_range[1])
        ]
    elif len(date_range) == 1:
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date == date_range[0])
        ]
    
    # Check for alerts
    alerts = check_for_alerts(filtered_df)
    if alerts:
        st.sidebar.header("Alerts")
        for alert in alerts:
            if alert["severity"] == "high":
                st.sidebar.error(f"🚨 {alert['message']}")
            else:
                st.sidebar.warning(f"⚠️ {alert['message']}")
    
    # KPI Cards
    #st.subheader("Key Performance Indicators")
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h2 style="color: #00843D; border-bottom: 2px solid #00843D; padding-bottom: 8px; display: inline-block;">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#00843D" style="vertical-align: middle; margin-right: 10px;">
                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14zM7 10h2v7H7zm4-3h2v10h-2zm4 6h2v4h-2z"/>
            </svg>
            Performance Overview
        </h2>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#00843D" style="margin-right: 8px;">
                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V5h14v14z"/>
                    <path d="M7 12h2v5H7zm4-7h2v12h-2zm4 5h2v7h-2z"/>
                </svg>
                <span style="font-weight: 600; color: #00843D;">Total Feedback</span>
            </div>
            <h2>{len(filtered_df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        positive_pct = len(filtered_df[filtered_df['sentiment']=='POSITIVE'])/len(filtered_df) if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#2ecc71" style="margin-right: 8px;">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                <span style="font-weight: 600; color: #004A8D;">Positive Sentiment</span>
            </div>
            <h2>{positive_pct:.0%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        negative_pct = len(filtered_df[filtered_df['sentiment']=='NEGATIVE'])/len(filtered_df) if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#e74c3c" style="margin-right: 8px;">
                    <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
                <span style="font-weight: 600; color: #004A8D;">Negative Sentiment</span>
            </div>
            <h2>{negative_pct:.0%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        critical_count = len(filtered_df[filtered_df['is_negative']])
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#f39c12" style="margin-right: 8px;">
                    <path d="M12 2L4 5v6.09c0 5.05 3.41 9.76 8 10.91 4.59-1.15 8-5.86 8-10.91V5l-8-3zm-1 15h2v2h-2zm0-10h2v6h-2z"/>
                </svg>
                <span style="font-weight: 600; color: #004A8D;">Critical Issues</span>
            </div>
            <h2>{critical_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview", 
        "📱 Channel Analysis", 
        "🌍 Regional View", 
        "🧠 Advanced Analytics", 
        "🔍 Anomalies", 
        "📈 Advanced Viz",
        "📝 Executive Report"
    ])
    
    with tab1:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color: #004A8D; margin-top: 0;">Pension Products Sentiment Overview</h3>
        """, unsafe_allow_html=True)
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.pie(filtered_df, names='sentiment', 
                 title='Sentiment Distribution',
                 color='sentiment',
                 color_discrete_map={'POSITIVE':'#00843D','NEGATIVE':'#D4122F','NEUTRAL':'#00A551'})
                fig1.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#003D2B")
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                daily_sentiment = filtered_df.set_index('timestamp').groupby([pd.Grouper(freq='D'), 'sentiment']).size().unstack()
                fig2 = px.line(daily_sentiment, 
                              title='Daily Sentiment Trend',
                              labels={'value':'Count', 'timestamp':'Date'},
                              color_discrete_map={'POSITIVE':'#2ecc71','NEGATIVE':'#e74c3c','NEUTRAL':'#3498db'})
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#004A8D")
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No data matches your filter criteria. Please adjust your filters.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Channel Performance for Pension Products")
        
        # Channel sentiment distribution
        fig3 = px.bar(filtered_df.groupby('channel')['sentiment'].value_counts(normalize=True).reset_index(name='percentage'), 
                     x='channel', y='percentage', color='sentiment',
                     title='Pension Products Sentiment Distribution by Channel',
                     barmode='group',
                     color_discrete_map={'POSITIVE':'#2ecc71','NEGATIVE':'#e74c3c','NEUTRAL':'#3498db'})
        st.plotly_chart(fig3, use_container_width=True)
        
        # Channel resolution time
        if 'resolution_time' in filtered_df.columns:
            st.subheader("Average Resolution Time by Channel")
            resolution_times = filtered_df[filtered_df['resolution_time'].notna()].groupby('channel')['resolution_time'].mean().reset_index()
            fig4 = px.bar(resolution_times, x='channel', y='resolution_time',
                         title='Average Resolution Time for Pension Products (minutes)',
                         labels={'resolution_time':'Minutes'})
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.subheader("Regional Analysis of Pension Products")
        
        # Regional sentiment heatmap
        fig5 = px.density_heatmap(filtered_df, x='region', y='product', 
                                 z='score', histfunc='avg',
                                 title='Average Sentiment Score by Region and Pension Product',
                                 color_continuous_scale='RdYlGn')
        st.plotly_chart(fig5, use_container_width=True)
        
        # Regional comparison
        st.subheader("Regional Comparison of Pension Products")
        fig6 = px.box(filtered_df, x='region', y='score', color='sentiment',
                     title='Pension Products Sentiment Score Distribution by Region',
                     color_discrete_map={'POSITIVE':'#2ecc71','NEGATIVE':'#e74c3c','NEUTRAL':'#3498db'})
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab4:
        st.subheader("🔍 Advanced Sentiment Analytics")
        st.markdown("""
        <div style="background-color: #E6F4F1; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <strong>AI-powered insights</strong> into emerging pension product trends
        </div>
        """, unsafe_allow_html=True)
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Sentiment Trend Forecasting")
                try:
                    # Prepare time series data with proper validation
                    if not filtered_df.empty:
                        daily_sentiment = filtered_df.set_index('timestamp')['score'].resample('D').mean()
                        
                        # Ensure we have enough data points
                        if len(daily_sentiment) > 5:  # Minimum 5 days for ARIMA(3,1,1)
                            model = ARIMA(daily_sentiment, order=(3,1,1))
                            model_fit = model.fit()
                            forecast = model_fit.get_forecast(steps=7)
                            
                            # Create visualization
                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                x=daily_sentiment.index,
                                y=daily_sentiment,
                                name='Historical',
                                line=dict(color='#00843D')
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                x=forecast.predicted_mean.index,
                                y=forecast.predicted_mean,
                                name='Forecast',
                                line=dict(color='#00A551', dash='dot')
                                )
                            )
                            fig.update_layout(
                                title='7-Day Sentiment Forecast',
                                yaxis_title='Sentiment Score',
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Insufficient data for forecasting (need at least 5 days)")
                    else:
                        st.warning("No data available for forecasting")
                except Exception as e:
                    st.warning(f"Forecasting error: {str(e)}")
            
            with col2:
                st.markdown("#### Topic Modeling")
                try:
                    # Prepare text data
                    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                    tfidf = vectorizer.fit_transform(filtered_df['feedback_text'])
                    
                    # Fit LDA model
                    lda = LatentDirichletAllocation(n_components=3, random_state=42)
                    lda.fit(tfidf)
                    
                    # Display topics
                    st.markdown("**Detected Themes in Feedback:**")
                    for idx, topic in enumerate(lda.components_):
                        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]
                        st.markdown(f"""
                        <div style="background-color: white; padding: 10px; border-radius: 5px; margin: 5px 0; 
                                    border-left: 3px solid #00843D">
                            <strong>Topic {idx+1}:</strong> {', '.join(top_words)}
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Topic modeling unavailable: {str(e)}")
        
        else:
            st.warning("No data available for analysis")
    
    with tab5:
        st.subheader("📈 Advanced Data Visualizations")
        st.markdown("""
        <div style="background-color: #E6F4F1; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <strong>Interactive visualizations</strong> for deeper insight exploration
        </div>
        """, unsafe_allow_html=True)
        
        if not filtered_df.empty:
            viz_type = st.selectbox("Choose Visualization", 
                                  ["Sankey Flow", "Sentiment Heatmap"])
            
            if viz_type == "Sankey Flow":
                st.markdown("#### Customer Journey Flow")
                
                # Prepare Sankey data
                source = []
                target = []
                value = []
                
                # Example mapping (adapt with real data relationships)
                nodes = ["Twitter", "Facebook", "Negative", "Positive", "GPP", "GSIPP"]
                source_indices = [0, 0, 1, 1, 2, 3]
                target_indices = [2, 3, 2, 3, 4, 5]
                values = [10, 30, 15, 40, 25, 35]
                
                fig = go.Figure(
                    go.Sankey(
                        node=dict(
                            label=nodes,
                            color=['#00843D']*2 + ['#D4122F', '#00A551', '#003D2B', '#00843D']
                        ),  # Close node dict
                        link=dict(
                            source=source_indices,
                            target=target_indices,
                            value=values
                        )  # Close link dict
                    )  # Close Sankey
                )  # Close Figure

                st.plotly_chart(fig, use_container_width=True)           
            
            
            elif viz_type == "Sentiment Heatmap":
                st.markdown("#### Temporal Sentiment Patterns")
                
                # Create time-based heatmap
                filtered_df['hour'] = filtered_df['timestamp'].dt.hour
                heatmap_data = filtered_df.pivot_table(
                    index='hour',
                    columns='product',
                    values='score',
                    aggfunc='mean'
                )
                
                fig = px.imshow(
                    heatmap_data,
                    labels=dict(x="Product", y="Hour", color="Sentiment"),
                    color_continuous_scale=['#D4122F', '#FFFFFF', '#00843D']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No data available for visualization")

    with tab6:
        st.subheader("🚨 Pension Product Anomaly Detection")
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <b>AI-powered detection</b> of unusual feedback patterns that may indicate emerging pension product issues
        </div>
        """, unsafe_allow_html=True)
        
        anomaly_df = detect_anomalies(filtered_df)
        
        if not anomaly_df.empty:
            st.metric("Unusual Pension Feedback Detected", len(anomaly_df), delta=f"{len(anomaly_df)/len(filtered_df):.1%} of total")
            
            fig_anom = px.scatter(
                anomaly_df,
                x='timestamp',
                y='score',
                color='sentiment',
                hover_data=['feedback_text'],
                title="Pension Product Anomaly Timeline",
                color_discrete_map={'POSITIVE':'#2ecc71','NEGATIVE':'#e74c3c'}
            )
            st.plotly_chart(fig_anom, use_container_width=True)
            
            st.subheader("Most Unusual Pension Product Feedback")
            for _, row in anomaly_df.head(5).iterrows():
                with st.expander(f"{row['channel']} - {row['region']} (Score: {row['score']:.2f})"):
                    st.write(row['feedback_text'])
                    st.caption(f"Pension Product: {row['product']} | {row['timestamp']}")
        else:
            st.success("No anomalies detected in current pension product dataset")

    with tab7:
        st.subheader("📊 Pension Products Executive Summary")
        st.markdown("""
        <div style="background-color: #e7f5fe; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <b>AI-generated insights</b> powered by Mistral-7B-v0.3 Large Language Model
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Summary Report", type="primary"):
            with st.spinner("Analyzing pension product trends with Mistral-7B..."):
                try:
                    summary = generate_executive_summary(filtered_df)
                    
                    st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-top: 20px;">
                        <h4 style="color: #1a5276;">Lloyds Pension Products Sentiment Brief</h4>
                        {summary.replace("\n", "<br>")}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    pdf_path = generate_summary_pdf(summary)
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    
                    st.download_button(
                        label="📄 Download Summary as PDF",
                        data=pdf_data,
                        file_name="lloyds_pension_sentiment_summary.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Failed to generate summary: {str(e)}")
                    st.info("Please check your Hugging Face API token and internet connection")
        else:
            st.info("Click the button above to generate an executive summary for pension products")


if __name__ == "__main__":  
    main()  
