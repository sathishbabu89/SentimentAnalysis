Here's the complete `pip install` command with all required packages for the UK sentiment analysis dashboard:

```bash
pip install pandas numpy streamlit plotly folium streamlit-folium transformers torch python-dotenv faker
```

### Package Breakdown:

1. **Core Data & Visualization**:
   - `pandas` - Data manipulation and analysis
   - `numpy` - Numerical operations
   - `plotly` - Interactive visualizations

2. **Dashboard Framework**:
   - `streamlit` - Web app framework
   - `streamlit-folium` - Map integration with Streamlit

3. **Mapping**:
   - `folium` - Interactive maps and heatmaps

4. **Sentiment Analysis**:
   - `transformers` - Hugging Face's NLP models
   - `torch` - PyTorch (required for transformers)

5. **Utilities**:
   - `python-dotenv` - Environment variable management
   - `faker` - Generating realistic test data

### Recommended Installation Steps:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate  # Windows
   ```

2. Run the complete install command:
   ```bash
   pip install pandas numpy streamlit plotly folium streamlit-folium transformers torch python-dotenv faker fpdf langchain_community langchain scikit-learn
   ```

3. Verify installations:
   ```bash
   pip list
   ```

4. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

This will install all dependencies needed for:
- UK-wide geographic visualization
- Interactive sentiment heatmaps
- Realistic data simulation
- The complete dashboard with all features we've implemented
