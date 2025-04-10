import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from data_simulator import DataSimulator
from london_map import create_london_sentiment_map
from uk_map import create_uk_sentiment_map
from streamlit_folium import folium_static
import numpy as np

# Initialize simulator
simulator = DataSimulator()

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
            "message": f"Product issue detected with {product_stats.idxmax()} ({(product_stats.max()*100):.1f}%)",
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

def main():
    st.set_page_config(
        page_title="Lloyds Sentiment Dashboard",
        layout="wide",
        page_icon="🏦"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .metric-card {
            background-color: #1a5276;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .critical-issue {
            border-left: 5px solid #e74c3c;
            padding-left: 10px;
        }
        .resolved-issue {
            border-left: 5px solid #2ecc71;
            padding-left: 10px;
        }
        .map-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🏦 Lloyds Banking Group - Customer Sentiment Dashboard")
    st.markdown("""
    <div style="color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px;">
    Real-time monitoring of customer sentiment across digital channels
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_channels = st.sidebar.multiselect(
        "Select Channels", 
        df['channel'].unique(), 
        default=df['channel'].unique()
    )
    selected_regions = st.sidebar.multiselect(
        "Select Regions",
        df['region'].unique(),
        default=df['region'].unique()
    )
    selected_products = st.sidebar.multiselect(
        "Select Products",
        df['product'].unique(),
        default=df['product'].unique()
    )
    
    # Time filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data
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
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">Total Feedback<br><h2>{:,}</h2></div>'.format(len(filtered_df)), 
                   unsafe_allow_html=True)
    
    with col2:
        positive_pct = len(filtered_df[filtered_df['sentiment']=='POSITIVE'])/len(filtered_df)
        st.markdown(f'<div class="metric-card">Positive Sentiment<br><h2>{positive_pct:.0%}</h2></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        negative_pct = len(filtered_df[filtered_df['sentiment']=='NEGATIVE'])/len(filtered_df)
        st.markdown(f'<div class="metric-card">Negative Sentiment<br><h2>{negative_pct:.0%}</h2></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        critical_count = len(filtered_df[filtered_df['is_negative']])
        st.markdown(f'<div class="metric-card">Critical Issues<br><h2>{critical_count}</h2></div>', 
                   unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Channel Analysis", "Regional View", "Regional Map", "Issue Management"])
    
    with tab1:
        st.subheader("Sentiment Overview")
        
        # Sentiment distribution
        fig1 = px.pie(filtered_df, names='sentiment', 
                      title='Overall Sentiment Distribution',
                      color='sentiment',
                      color_discrete_map={'POSITIVE':'#2ecc71','NEGATIVE':'#e74c3c','NEUTRAL':'#3498db'})
        st.plotly_chart(fig1, use_container_width=True)
        
        # Sentiment trend
        st.subheader("Sentiment Trend Over Time")
        daily_sentiment = filtered_df.set_index('timestamp').groupby([pd.Grouper(freq='D'), 'sentiment']).size().unstack()
        fig2 = px.line(daily_sentiment, 
                      title='Daily Sentiment Trend',
                      labels={'value':'Count', 'timestamp':'Date'},
                      color_discrete_map={'POSITIVE':'#2ecc71','NEGATIVE':'#e74c3c','NEUTRAL':'#3498db'})
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Channel Performance")
        
        # Channel sentiment distribution
        fig3 = px.bar(filtered_df.groupby('channel')['sentiment'].value_counts(normalize=True).reset_index(name='percentage'), 
                     x='channel', y='percentage', color='sentiment',
                     title='Sentiment Distribution by Channel',
                     barmode='group',
                     color_discrete_map={'POSITIVE':'#2ecc71','NEGATIVE':'#e74c3c','NEUTRAL':'#3498db'})
        st.plotly_chart(fig3, use_container_width=True)
        
        # Channel resolution time
        if 'resolution_time' in filtered_df.columns:
            st.subheader("Average Resolution Time by Channel")
            resolution_times = filtered_df[filtered_df['resolution_time'].notna()].groupby('channel')['resolution_time'].mean().reset_index()
            fig4 = px.bar(resolution_times, x='channel', y='resolution_time',
                         title='Average Resolution Time (minutes)',
                         labels={'resolution_time':'Minutes'})
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.subheader("Regional Analysis")
        
        # Regional sentiment heatmap
        fig5 = px.density_heatmap(filtered_df, x='region', y='product', 
                                 z='score', histfunc='avg',
                                 title='Average Sentiment Score by Region and Product',
                                 color_continuous_scale='RdYlGn')
        st.plotly_chart(fig5, use_container_width=True)
        
        # Regional comparison
        st.subheader("Regional Comparison")
        fig6 = px.box(filtered_df, x='region', y='score', color='sentiment',
                     title='Sentiment Score Distribution by Region',
                     color_discrete_map={'POSITIVE':'#2ecc71','NEGATIVE':'#e74c3c','NEUTRAL':'#3498db'})
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab4:
        st.subheader("UK Regional Analysis")
        
        if not filtered_df.empty:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                <p style="margin: 0;">🔴 <strong>Red areas</strong> show negative sentiment hotspots</p>
                <p style="margin: 0;">🟢 <strong>Green markers</strong> indicate positive feedback locations</p>
            </div>
            """, unsafe_allow_html=True)
            
            # UK Map
            uk_map = create_uk_sentiment_map(filtered_df)
            folium_static(uk_map, width=800, height=600)
            
            # Regional statistics
            st.subheader("Regional Sentiment Metrics")
            regional_stats = filtered_df.groupby('region').agg({
                'sentiment': lambda x: (x == 'POSITIVE').mean(),
                'score': 'mean',
                'is_negative': 'sum'
            }).sort_values('is_negative', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(regional_stats, 
                            x=regional_stats.index, 
                            y='sentiment',
                            title='Positive Sentiment by Region')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(regional_stats, 
                            x=regional_stats.index, 
                            y='is_negative',
                            title='Critical Issues by Region')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available with current filters")
    
    with tab5:
        st.subheader("Issue Management Dashboard")
        
        # Critical issues table
        critical_issues = filtered_df[filtered_df['is_negative']].sort_values('score', ascending=False)
        
        if not critical_issues.empty:
            st.markdown(f"**{len(critical_issues)} Critical Issues Identified**")
            
            for _, row in critical_issues.head(10).iterrows():
                container = st.container(border=True)
                with container:
                    cols = st.columns([3,1,1,1])
                    cols[0].markdown(f"**{row['channel']}** - {row['region']}")
                    cols[1].metric("Score", f"{row['score']:.2f}")
                    cols[2].metric("Product", row['product'])
                    
                    if pd.notna(row['resolution_status']):
                        cols[3].success(f"Status: {row['resolution_status']}")
                    else:
                        cols[3].error("Status: Unresolved")
                    
                    st.markdown(f"*{row['feedback_text']}*")
                    
                    if pd.notna(row['resolution_notes']):
                        with st.expander("Resolution Notes"):
                            st.info(row['resolution_notes'])
                    
                    if st.button("Take Action", key=row['customer_id']):
                        st.session_state[f"action_{row['customer_id']}"] = True
                    
                    if st.session_state.get(f"action_{row['customer_id']}", False):
                        action = st.selectbox(
                            "Select action",
                            ["Contact customer", "Escalate to manager", "Mark as resolved"],
                            key=f"action_select_{row['customer_id']}"
                        )
                        if st.button("Confirm Action", key=f"confirm_{row['customer_id']}"):
                            st.success(f"Action '{action}' taken for issue ID: {row['customer_id']}")
                            st.session_state[f"action_{row['customer_id']}"] = False
        else:
            st.success("No critical issues identified in the selected filters")

if __name__ == "__main__":
    main()
