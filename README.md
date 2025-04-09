# LBG Sentiment Analysis POC

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Lloyds+Sentiment+Dashboard+Screenshot)

## Overview

This Proof of Concept (POC) demonstrates a real-time sentiment analysis dashboard tailored for Lloyds Banking Group's digital customer engagement channels. The system monitors customer feedback across multiple platforms, analyzes sentiment, identifies critical issues, and enables proactive resolution.

## Key Features

- Real-time sentiment analysis of customer feedback
- Multi-channel monitoring (social media, app reviews, call transcripts)
- Regional sentiment heatmaps
- Automated alerting for critical issues
- Actionable insights for customer service teams
- Performance tracking by product and channel

## Components

### 1. Sentiment Analysis Engine (`sentiment_analyzer.py`)

**Purpose**: Performs the core sentiment analysis using NLP models.

**Key Aspects**:
- Uses Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` model
- Processes text to determine sentiment (Positive/Negative/Neutral)
- Calculates sentiment confidence scores
- Flags highly negative feedback for escalation
- Includes error handling and logging

**Implementation Details**:
- Wrapper around Hugging Face pipeline
- Text truncation to handle long feedback
- Score thresholding for critical issues
- Graceful fallback for analysis failures

### 2. Data Simulator (`data_simulator.py`)

**Purpose**: Generates realistic synthetic banking customer feedback data.

**Key Aspects**:
- Simulates multiple channels (Twitter, Facebook, App Reviews, etc.)
- Covers different UK regions
- Includes various banking products
- Generates realistic feedback text
- Simulates resolution timelines and notes

**Implementation Details**:
- Uses Faker library for realistic data generation
- Balanced distribution of positive/negative/neutral feedback
- Includes timestamps for temporal analysis
- Generates resolution data for negative feedback
- Maintains consistent customer IDs

### 3. Dashboard Application (`dashboard.py`)

**Purpose**: Interactive visualization and management interface.

**Key Features**:

#### Filtering System
- Channel selection (Twitter, Facebook, etc.)
- Regional filters
- Product type filters
- Time period selection

#### Visualization Components
- Real-time sentiment distribution charts
- Time-series trend analysis
- Regional heatmaps
- Channel performance comparisons
- Product-specific sentiment analysis

#### Alerting System
- Automatic detection of sentiment spikes
- Regional alert notifications
- Product-specific issue identification
- Channel performance warnings

#### Issue Management
- Critical issues prioritization
- Action tracking
- Resolution status updates
- Team assignment workflow

### 4. Configuration (`requirements.txt`)

Lists all Python dependencies with version constraints to ensure consistent environment setup.

## How It Works

### Data Flow

1. **Data Generation**:
   - The Data Simulator creates synthetic but realistic customer feedback
   - Feedback includes channel, region, product, and timestamp metadata

2. **Sentiment Analysis**:
   - Each feedback item is processed by the Sentiment Analyzer
   - Sentiment label and score are attached to each record
   - Highly negative feedback is flagged for review

3. **Dashboard Processing**:
   - Data is loaded into the Streamlit application
   - Filters are applied based on user selection
   - Visualizations are generated in real-time
   - Alerts are triggered based on analysis

4. **User Interaction**:
   - Customer service teams view the dashboard
   - Critical issues are identified and assigned
   - Resolution actions are tracked
   - Trends are monitored over time

### Architecture

```
┌────────────────┐    ┌───────────────────┐    ┌────────────────────┐
│ Data Simulator │───▶│ Sentiment Analyzer │───▶│ Streamlit Dashboard │
└────────────────┘    └───────────────────┘    └────────────────────┘
      │                      │                           │
      │  Synthetic           │  Sentiment                │  Visualizations
      │  Feedback Data       │  Scores & Labels          │  Alerts & Actions
      ▼                      ▼                           ▼
┌────────────────┐    ┌───────────────────┐    ┌────────────────────┐
│  Sample Data   │    │  Analyzed Data    │    │  User Interface    │
│  (CSV/JSON)    │    │  (Enriched)       │    │  (Web Browser)     │
└────────────────┘    └───────────────────┘    └────────────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip package manager
- Git (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/lloyds-sentiment-poc.git
   cd lloyds-sentiment-poc
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

The application will launch in your default browser at `http://localhost:8501`.

## Usage Guide

1. **Dashboard Overview**:
   - The main view shows key metrics and sentiment distribution
   - Use the sidebar filters to focus on specific channels, regions, or time periods

2. **Identifying Issues**:
   - Critical issues are highlighted in the Alerts section
   - The Issues tab shows all negative feedback requiring attention
   - Sort by sentiment score to prioritize the most severe cases

3. **Taking Action**:
   - Select an issue to view details
   - Choose an action from the dropdown (Contact customer, Escalate, etc.)
   - Click "Confirm Action" to log your response

4. **Analyzing Trends**:
   - Use the time period filter to analyze trends
   - Compare performance across regions and channels
   - Monitor resolution times for negative feedback

## Customization Options

1. **Real Data Integration**:
   - Replace `data_simulator.py` with actual data connectors
   - Possible sources: Twitter API, App Store reviews, CRM systems

2. **Model Customization**:
   - Fine-tune the sentiment model on banking-specific language
   - Add domain-specific categories (fraud, service quality, etc.)

3. **Alert Thresholds**:
   - Adjust sentiment score thresholds in `sentiment_analyzer.py`
   - Modify alert logic in the dashboard's `check_for_alerts` function

4. **Visualizations**:
   - Add new chart types in `dashboard.py`
   - Customize color schemes and layouts

## Roadmap

- [ ] Integration with real data sources
- [ ] Custom model fine-tuning on banking terminology
- [ ] Team assignment and workflow integration
- [ ] SLA tracking for issue resolution
- [ ] Automated report generation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
