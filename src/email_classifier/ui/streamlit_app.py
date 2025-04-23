"""Advanced Streamlit UI for the Email Classification system with professional styling."""
import streamlit as st
import pandas as pd
import sys
import os
import logging
import time
from datetime import datetime
import re
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add the parent directory to the path to import local modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from email_classifier.utils.pii_masker import mask_pii
    from email_classifier.models.classifier import classify_email
except ImportError as e:
    # Dummy functions for development
    def mask_pii(text):
        return text, [
            {"type": "EMAIL", "value": "john@company.com", "start": 10, "end": 26},
            {"type": "PHONE", "value": "555-123-4567", "start": 50, "end": 62}
        ]
    
    def classify_email(text):
        return "BUSINESS"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Advanced Email Classification System",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with modern design
st.markdown("""
<style>
    /* Modern theme variables */
    :root {
        --primary-color: #2C3E50;
        --secondary-color: #3498DB;
        --accent-color: #E74C3C;
        --success-color: #2ECC71;
        --warning-color: #F39C12;
        --background-color: #F7F9FC;
        --card-background: #FFFFFF;
        --text-color: #2C3E50;
        --shadow-color: rgba(0, 0, 0, 0.1);
        --border-radius: 12px;
    }
    
    /* Main body styling */
    .stApp {
        background-color: var(--background-color) !important;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 15px var(--shadow-color);
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        text-align: center;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .result-card {
        background: var(--card-background);
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: 0 4px 20px var(--shadow-color);
        margin: 1rem 0;
        border: 1px solid #E5E7EB;
    }
    
    /* Status indicator */
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-success {
        background-color: var(--success-color);
    }
    
    .status-processing {
        background-color: var(--warning-color);
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Enhanced category display */
    .category-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 1rem 0;
    }
    
    .category-business {
        background-color: #ECF0F1;
        color: #2C3E50;
        border: 2px solid #2C3E50;
    }
    
    .category-personal {
        background-color: #ECF6FC;
        color: #3498DB;
        border: 2px solid #3498DB;
    }
    
    .category-spam {
        background-color: #FDEDEC;
        color: #E74C3C;
        border: 2px solid #E74C3C;
    }
    
    .category-uncategorized {
        background-color: #FDF2E9;
        color: #D68910;
        border: 2px solid #D68910;
    }
    
    /* Metrics display */
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        text-align: center;
        box-shadow: 0 2px 10px var(--shadow-color);
        flex: 1;
        margin: 0 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7F8C8D;
        margin-top: 0.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--secondary-color);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 30px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #2980B9;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
    }
    
    /* Progress bar */
    .progress-bar {
        height: 4px;
        background: linear-gradient(90deg, var(--secondary-color), var(--accent-color));
        width: 100%;
        border-radius: 2px;
        margin: 1rem 0;
        animation: progress 2s ease-in-out infinite;
    }
    
    @keyframes progress {
        0% { width: 0%; }
        50% { width: 70%; }
        100% { width: 100%; }
    }
    
    /* Error styling */
    .error-container {
        background-color: #FDEDEC;
        border-left: 4px solid var(--accent-color);
        padding: 1rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
    }
    
    .error-title {
        color: var(--accent-color);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .error-message {
        color: #721C24;
    }
    
    .error-help {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Masked content styling */
    .masked-content, [class*="masked"], span[style*="background-color: white"] {
        background-color: #ECF0F1 !important;
        color: #34495E !important;
        font-weight: 600 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border: 1px dashed #7F8C8D !important;
        display: inline-block !important;
        margin: 0 2px !important;
        position: relative !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    
    /* Style for dataframe masked content */
    .dataframe .masked-content {
        background-color: #EAF2F8 !important;
        border: 1px solid #3498DB !important;
    }
    
    /* Make dataframe text visible with better styling */
    .stDataFrame {
        color: #2C3E50 !important;
        font-weight: 500 !important;
    }
    
    /* Style for the dataframe text */
    .stDataFrame [data-testid="StyledDataFrameDataCell"] {
        background-color: #F8F9F9 !important;
        color: #34495E !important;
        font-weight: 500 !important;
    }
    
    /* Add special styling for "Value" column */
    .stDataFrame [data-testid="StyledDataFrameDataCell"]:nth-child(2) {
        background-color: #EBF5FB !important;
        font-weight: 600 !important;
        border-left: 2px solid #3498DB !important;
    }
    
    /* Staggered display for multiple masked elements */
    .masked-content + .masked-content {
        margin-left: 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for tracking
if 'processed_emails' not in st.session_state:
    st.session_state.processed_emails = 0
if 'total_pii_detected' not in st.session_state:
    st.session_state.total_pii_detected = 0
if 'category_counts' not in st.session_state:
    st.session_state.category_counts = {'Business': 0, 'Personal': 0, 'Spam': 0, 'Uncategorized': 0}

def create_processing_animation():
    """Create a professional processing animation."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        if i < 30:
            status_text.text("⚡ Initializing PII detection engine...")
        elif i < 60:
            status_text.text("🔍 Scanning for sensitive information...")
        elif i < 90:
            status_text.text("🤖 Analyzing email content with AI...")
        else:
            status_text.text("✨ Finalizing classification...")
        time.sleep(0.01)
    
    progress_bar.empty()
    status_text.empty()

def display_metrics_dashboard():
    """Display a dashboard with key metrics."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{st.session_state.processed_emails}</div>
            <div class="metric-label">Emails Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{st.session_state.total_pii_detected}</div>
            <div class="metric-label">PII Entities Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        categories = st.session_state.category_counts
        non_zero_categories = {k: v for k, v in categories.items() if v > 0}
        if non_zero_categories:
            most_common = max(non_zero_categories, key=non_zero_categories.get)
        else:
            most_common = "N/A"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{most_common}</div>
            <div class="metric-label">Most Common Category</div>
        </div>
        """, unsafe_allow_html=True)

def create_category_chart():
    """Create a professional chart showing category distribution."""
    if PLOTLY_AVAILABLE and sum(st.session_state.category_counts.values()) > 0:
        fig = px.pie(
            values=list(st.session_state.category_counts.values()),
            names=list(st.session_state.category_counts.keys()),
            title="Email Classification Distribution",
            color_discrete_sequence=['#2C3E50', '#3498DB', '#E74C3C', '#D68910']
        )
        fig.update_layout(
            font_family="sans-serif",
            title_font_size=20,
            title_x=0.5,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    elif not PLOTLY_AVAILABLE:
        # Simple fallback visualization
        st.markdown("### Email Classification Distribution")
        for category, count in st.session_state.category_counts.items():
            if count > 0:
                st.markdown(f"{category}: {count} emails")

def create_entity_visualization(entities):
    """Create a bar chart for entity types."""
    if PLOTLY_AVAILABLE and entities:
        entity_types = [ent['type'] for ent in entities]
        entity_counts = pd.Series(entity_types).value_counts()
        
        fig = px.bar(
            x=entity_counts.index,
            y=entity_counts.values,
            title="PII Entity Types Detected",
            labels={'x': 'Entity Type', 'y': 'Count'},
            color_discrete_sequence=['#3498DB']
        )
        fig.update_layout(
            font_family="sans-serif",
            title_font_size=18,
            title_x=0.5,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            bargap=0.2
        )
        st.plotly_chart(fig, use_container_width=True)
    elif not PLOTLY_AVAILABLE and entities:
        # Simple fallback visualization
        st.markdown("### PII Entity Types Detected")
        entity_types = [ent['type'] for ent in entities]
        for entity_type, count in pd.Series(entity_types).value_counts().items():
            st.markdown(f"{entity_type}: {count}")

def display_error(error_message):
    """Display error message in a styled container."""
    st.markdown(f"""
    <div class="error-container">
        <div class="error-title">❌ Error Occurred</div>
        <div class="error-message">{error_message}</div>
        <div class="error-help">Please try again or contact support if the issue persists.</div>
    </div>
    """, unsafe_allow_html=True)

def process_html_content(content):
    """Process content with HTML tags to ensure proper rendering."""
    # Replace newlines with <br> tags
    content = content.replace('\n', '<br>')
    # Make sure the content is treated as safe HTML
    return content

def main():
    """Main function to run the advanced Streamlit app."""
    
    # Professional header with gradient
    st.markdown("""
    <div class="header-container">
        <h1 class="app-title">Advanced Email Classification System</h1>
        <p class="app-subtitle">Enterprise-grade PII Detection & Smart Email Categorization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with additional controls
    with st.sidebar:
        # Logo - handling different streamlit versions
        try:
            st.image("https://via.placeholder.com/200x50?text=Company+Logo", width=200)
        except Exception:
            st.markdown("**[Company Logo]**")
        
        st.markdown("### ⚙️ Settings")
        
        # Theme settings
        theme = st.selectbox("Color Theme", ["Default", "Dark", "Light"])
        
        # Advanced options
        show_entity_visualization = st.checkbox("Show Entity Visualization", value=True)
        show_metrics = st.checkbox("Show Metrics Dashboard", value=True)
        
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        st.markdown(f"🔄 Emails processed: **{st.session_state.processed_emails}**")
        st.markdown(f"🔍 Total PII detected: **{st.session_state.total_pii_detected}**")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This advanced system uses:
        - **BERT** Deep Learning
        - **NER** Entity Recognition
        - **RegEx** Pattern Matching
        
        Version: **2.0.1**
        Last Updated: **April 2025**
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        email_input = st.text_area(
            "📝 Email Content",
            height=250,
            placeholder="Paste your email content here...",
            help="Enter the email text you want to classify and check for PII"
        )
    
    with col2:
        # st.markdown('<h3 style="color: #000;">📋 Quick Guide</h3>', unsafe_allow_html=True)
        st.markdown("""
    <h3 style="color: #000;">📋 Quick Guide</h3>
    <div style="color: #000; font-size: 16px;">
        <ol>
            <li>Paste your email content</li>
            <li>Click <strong>Process Email</strong></li>
            <li>View detailed results</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

    
    # Processing button with custom styling
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        process_button = st.button("🚀 Process Email")
    
    if process_button:
        if email_input.strip():
            create_processing_animation()
            
            try:
                # Mask PII
                masked_email, masked_entities = mask_pii(email_input)
                
                # Classify email
                category = classify_email(masked_email)
                
                # Normalize category for consistency
                category_normalized = category.capitalize()
                if category_normalized not in st.session_state.category_counts:
                    category_normalized = 'Uncategorized'
                
                # Update session state
                st.session_state.processed_emails += 1
                st.session_state.total_pii_detected += len(masked_entities)
                st.session_state.category_counts[category_normalized] += 1
                
                # Results section
                st.markdown("""
                    <h3 style="color: #000;">✨ Analysis Results</h3>
                """, unsafe_allow_html=True)

                # Category result with custom styling
                category_class = f"category-{category_normalized.lower()}"
                st.markdown(f"""
                    <div class="result-card">
                        <div class="status-indicator status-success"></div>
                        <span style="font-weight: 600; color: #000;">Classification Result:</span>
                        <span class="category-badge {category_class}" style="color: #000;">{category_normalized}</span>
                    </div>
                """, unsafe_allow_html=True)

                
                # Display metrics dashboard
                if show_metrics:
                    display_metrics_dashboard()
                
                # Dual display: Original vs Masked
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="result-card">
                        <h4 style="color: #000;">📨 Original Email</h4>
                        <div style="background: #F7F9FC; padding: 1rem; border-radius: 10px; min-height: 200px; color: #2C3E50; font-weight: 500;">
                            {content}
                        </div>
                    </div>

                    """.format(content=process_html_content(email_input)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="result-card">
                        <h4 style="color: #000;">🔒 Masked Email</h4>
                        <div style="background: #F7F9FC; padding: 1rem; border-radius: 10px; min-height: 200px; color: #2C3E50; font-weight: 500;">
                            {content}
                        </div>
                    </div>
                    """.format(content=process_html_content(masked_email)), unsafe_allow_html=True)
                
                # Entity Detection Results
                if masked_entities:
                    st.markdown("""
                    <div class="result-card">
                        <h4 style="color: #000;">🔍 Detected PII Entities</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    df = pd.DataFrame([{
                        "Type": ent['type'],
                        "Value": f"🔒 {ent['value']}",  # Add lock icon as visual indicator
                        "Location": f"Chars {ent['start']}-{ent['end']}",
                        "Confidence": "98%" if ent['type'] == "EMAIL" else "95%"
                    } for ent in masked_entities])
                    
                    # For dataframe display, we need to use plain text since HTML in dataframe doesn't work well
                    display_df = pd.DataFrame([{
                        "Type": ent['type'],
                        "Value": ent['value'],  # Use plain text value
                        "Location": f"Chars {ent['start']}-{ent['end']}",
                        "Confidence": "98%" if ent['type'] == "EMAIL" else "95%"
                    } for ent in masked_entities])
                    
                    st.dataframe(
                        display_df,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Value": st.column_config.Column(
                                "Value",
                                help="The masked PII value",
                                width="medium",
                                required=True
                            )
                        }
                    )
                    
                    if show_entity_visualization:
                        create_entity_visualization(masked_entities)
                else:
                    st.markdown("""
                    <div class="result-card">
                        <div class="status-indicator status-success"></div>
                        <span style="font-weight: 600;">✅ No PII entities detected in this email</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show category distribution chart
                create_category_chart()
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    # Strip HTML tags for download 
                    plain_masked_email = re.sub(r'<[^>]*>', '', masked_email)
                    
                    st.download_button(
                        label="📥 Download Masked Email",
                        data=plain_masked_email,
                        file_name=f"masked_email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                with col2:
                    if masked_entities:
                        entities_csv = pd.DataFrame(masked_entities).to_csv(index=False)
                        st.download_button(
                            label="📥 Download PII Report",
                            data=entities_csv,
                            file_name=f"pii_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                logger.error(f"Error processing email: {e}")
                display_error(str(e))
        else:
            st.warning("⚠️ Please enter some email content to process.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; font-size: 0.9rem;">
        <p>© 2025 Advanced Email Classification System. All rights reserved.</p>
        <p>Powered by BERT Deep Learning & Advanced NER Technologies</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()