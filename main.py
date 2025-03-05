import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from fpdf import FPDF
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Enhanced Custom CSS for Modern, Sleek Design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        background-color: #f4f6f9;
        color: #2c3e50;
    }
    
    /* App Container */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e6e9f0 100%);
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Sidebar Enhancements */
    .css-1aumxhk {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
    }
    
    /* Header Styling */
    h1, h2, h3 {
        font-weight: 700;
        color: #1a73e8;
        letter-spacing: -0.5px;
    }
    
    /* Button Styling */
    .stButton button {
        background-color: #1a73e8 !important;
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 10px 20px;
        border: none;
    }
    .stButton button:hover {
        background-color: #185fcc !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(26,115,232,0.3);
    }
    
    /* Card Styling */
    .card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        border-left: 4px solid #1a73e8;
    }
    .card:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 25px rgba(0,0,0,0.12);
    }
    
    /* Typography */
    .stMarkdown {
        color: #34495e;
        line-height: 1.7;
        font-size: 16px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        color: #1a73e8;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #185fcc;
    }
    
    /* Input Styling */
    .stTextInput input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 10px;
    }
    .stTextInput input:focus {
        border-color: #1a73e8;
        box-shadow: 0 0 0 2px rgba(26,115,232,0.2);
    }
    
    /* Plotly Chart Styling */
    .plotly-graph-div {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced Data with More Comprehensive Information
AGENTS_DATA = {
    "Agent Name": ["Rajesh Kumar", "Priya Sharma", "Anil Desai", "Sneha Reddy"],
    "Language": ["Hindi", "English", "Marathi", "Telugu"],
    "Price Range": ["1-1.5 Cr", "80L-1 Cr", "1.5-2 Cr", "2.5-3 Cr"],
    "Specialization": ["Residential", "Commercial", "Luxury", "Investment Properties"],
    "Preferred Locations": ["City Center", "Near Schools", "Outskirts", "Near IT Park"],
    "Experience": ["5 years", "8 years", "10 years", "7 years"],
    "Client Satisfaction": [95, 92, 90, 94],
    "Deals Closed": [45, 62, 38, 55],
    "Average Commission": ["2.5%", "2.2%", "3%", "2.8%"],
    "Contact": ["rajesh.kumar@example.com", "priya.sharma@example.com", "anil.desai@example.com", "sneha.reddy@example.com"],
    "Testimonials": [
        "Rajesh helped us find the perfect home in the city center. Highly recommended!",
        "Priya is very professional and knows the market well.",
        "Anil has a great eye for luxury properties.",
        "Sneha is the go-to agent for high-end properties near IT parks."
    ],
    "Recent Deals": [
        "3BHK Apartment in City Center for 1.2 Cr",
        "2BHK Apartment near Schools for 90L",
        "Luxury Farmhouse on Outskirts for 1.8 Cr",
        "High-end Apartment near IT Park for 2.7 Cr"
    ],
    "Key Points": [
        "Focuses on 3BHK apartments in the city center.",
        "Specializes in 2BHK apartments near schools.",
        "Deals with luxury farmhouses on the outskirts.",
        "Expert in high-end properties near IT parks.",
    ],
}

# Convert to DataFrame
agents_df = pd.DataFrame(AGENTS_DATA)

# Function to generate agent report in Markdown format
def generate_agent_report(agent_name, df):
    agent_data = df[df['Agent Name'] == agent_name].iloc[0]
    report = f"""
    # Agent Report: {agent_name}
    ----------------------------
    
    ## 📝 **Overview**
    - **Language**: {agent_data['Language']}
    - **Specialization**: {agent_data['Specialization']}
    - **Experience**: {agent_data['Experience']}
    - **Client Satisfaction**: {agent_data['Client Satisfaction']}%
    - **Deals Closed**: {agent_data['Deals Closed']}
    - **Average Commission**: {agent_data['Average Commission']}
    - **Contact**: {agent_data['Contact']}
    
    ## 🌟 **Key Points**
    - {agent_data['Key Points']}
    
    ## 🏆 **Recent Deals**
    - {agent_data['Recent Deals']}
    
    ## 🗣️ **Client Testimonials**
    - "{agent_data['Testimonials']}"
    
    ## 📊 **Performance Summary**
    - **Client Satisfaction**: {agent_data['Client Satisfaction']}%
    - **Deals Closed**: {agent_data['Deals Closed']}
    - **Average Commission**: {agent_data['Average Commission']}
    """
    return report

# Function to generate PDF report
def generate_pdf_report(agent_name, df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    agent_data = df[df['Agent Name'] == agent_name].iloc[0]
    pdf.cell(200, 10, txt=f"Agent Report: {agent_name}", ln=True, align='C')
    pdf.ln(10)
    
    # Add sections to PDF
    sections = [
        ("Overview", f"""
        Language: {agent_data['Language']}
        Specialization: {agent_data['Specialization']}
        Experience: {agent_data['Experience']}
        Client Satisfaction: {agent_data['Client Satisfaction']}%
        Deals Closed: {agent_data['Deals Closed']}
        Average Commission: {agent_data['Average Commission']}
        Contact: {agent_data['Contact']}
        """),
        ("Key Points", agent_data['Key Points']),
        ("Recent Deals", agent_data['Recent Deals']),
        ("Client Testimonials", agent_data['Testimonials']),
    ]
    
    for section, content in sections:
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(200, 10, txt=section, ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=content)
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin1')

# Function to create agent performance chart
def create_agent_performance_chart(df):
    fig = px.bar(df, x='Agent Name', y='Deals Closed', color='Client Satisfaction',
                 title='Agent Performance: Deals Closed vs Client Satisfaction',
                 labels={'Deals Closed': 'Deals Closed', 'Client Satisfaction': 'Client Satisfaction (%)'})
    return fig

# Generalized Suggestions Data
GENERALIZED_SUGGESTIONS = [
    {
        "title": "Personalized Communication",
        "description": "Tailor your communication to match the client's preferences and needs.",
        "impact": "Increased client satisfaction and trust.",
        "key_techniques": [
            "Use client's preferred language.",
            "Focus on properties that match client's budget and preferences.",
            "Provide regular updates and follow-ups."
        ]
    },
    {
        "title": "Leverage Technology",
        "description": "Use advanced tools and platforms to enhance client interactions.",
        "impact": "Improved efficiency and client engagement.",
        "key_techniques": [
            "Use CRM tools to manage client relationships.",
            "Implement virtual tours for remote clients.",
            "Utilize data analytics for market insights."
        ]
    },
    {
        "title": "Focus on Client Education",
        "description": "Educate clients about the market trends and investment opportunities.",
        "impact": "Empowered clients make informed decisions.",
        "key_techniques": [
            "Provide market reports and insights.",
            "Host webinars and workshops.",
            "Share success stories and case studies."
        ]
    }
]

# Function to generate suggestions report
def generate_suggestions_report(suggestions):
    report = "# Strategic Communication Tactics Report\n\n"
    for suggestion in suggestions:
        report += f"## {suggestion['title']}\n"
        report += f"{suggestion['description']}\n"
        report += f"**Expected Impact:** {suggestion['impact']}\n"
        report += "**Key Implementation Techniques:**\n"
        for technique in suggestion['key_techniques']:
            report += f"- {technique}\n"
        report += "\n"
    return report

# Market Sentiment Analysis Function
def market_sentiment_analysis(df):
    # Calculate weighted sentiment score
    df['sentiment_score'] = (
        df['Client Satisfaction'] * 0.6 + 
        (df['Deals Closed'] / df['Deals Closed'].max() * 100) * 0.4
    )
    
    # Create pie chart of market sentiment
    fig = px.pie(
        df, 
        values='sentiment_score', 
        names='Agent Name', 
        title='Agent Market Sentiment Distribution',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    return fig

# Advanced Property Price Prediction Simulation
def simulate_property_price_trends():
    # Generate synthetic time series data
    np.random.seed(42)
    years = 5
    base_price = 1000000
    yearly_growth_rates = np.random.normal(0.07, 0.02, years)
    
    prices = [base_price]
    for rate in yearly_growth_rates:
        prices.append(prices[-1] * (1 + rate))
    
    # Create line plot with prediction intervals
    years_range = list(range(2022, 2022 + years + 1))
    
    # Confidence interval calculation
    confidence_level = 0.95
    confidence_interval = stats.t.interval(
        confidence_level, 
        len(prices) - 1, 
        loc=np.mean(prices), 
        scale=stats.sem(prices)
    )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years_range, 
        y=prices, 
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#1a73e8', width=3)
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=years_range + years_range[::-1],
        y=list(confidence_interval[0] * np.ones(len(years_range))) + 
          list(confidence_interval[1] * np.ones(len(years_range)))[::-1],
        fill='toself',
        fillcolor='rgba(26, 115, 232, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    
    fig.update_layout(
        title='Property Price Trend Prediction with 95% Confidence Interval',
        xaxis_title='Year',
        yaxis_title='Property Price (₹)',
        template='plotly_white'
    )
    
    return fig

# Machine Learning Feature Importance
def feature_importance_visualization(df):
    # Select numeric columns
    numeric_cols = ['Client Satisfaction', 'Deals Closed']
    X = df[numeric_cols]
    y = df['Client Satisfaction']
    
    # Use simple linear regression for feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Create feature importance plot
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=True)
    
    sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
    plt.title('Feature Importance in Client Satisfaction', fontsize=15)
    plt.xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    
    return plt

# Main App 
def main():
    st.title("🏡 Real Estate Intelligence Platform")
    st.markdown("**Empowering Agents with Data-Driven Insights**")

    # Add a new tab for Advanced Analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Agent Insights", 
        "Performance Dashboard", 
        "Strategic Suggestions", 
        "Market Sentiment", 
        "Predictive Analytics"
    ])

    with tab1:
        st.header("👥 Our Expert Agents")
        
        # Search bar for agents
        search_query = st.text_input("🔍 Search for an agent by name or specialization:", key="search")
        filtered_agents = agents_df[
            (agents_df['Agent Name'].str.contains(search_query, case=False)) |
            (agents_df['Specialization'].str.contains(search_query, case=False))
        ]
        
        if not filtered_agents.empty:
            selected_agent = st.selectbox("Select an agent to view details:", filtered_agents['Agent Name'])
            agent_data = filtered_agents[filtered_agents['Agent Name'] == selected_agent].iloc[0]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="card">
                    <h3>{agent_data['Agent Name']}</h3>
                    <p><strong>Language:</strong> {agent_data['Language']}</p>
                    <p><strong>Specialization:</strong> {agent_data['Specialization']}</p>
                    <p><strong>Experience:</strong> {agent_data['Experience']}</p>
                    <p><strong>Client Satisfaction:</strong> {agent_data['Client Satisfaction']}%</p>
                    <p><strong>Deals Closed:</strong> {agent_data['Deals Closed']}</p>
                    <p><strong>Average Commission:</strong> {agent_data['Average Commission']}</p>
                    <p><strong>Contact:</strong> {agent_data['Contact']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card">
                    <h3>About {agent_data['Agent Name']}</h3>
                    <p><strong>Testimonials:</strong> {agent_data['Testimonials']}</p>
                    <p><strong>Recent Deals:</strong> {agent_data['Recent Deals']}</p>
                    <p><strong>Key Points:</strong> {agent_data['Key Points']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Download agent report
            st.markdown("---")
            st.markdown("### 📥 Download Agent Report")
            col1, col2 = st.columns(2)
            
            with col1:
                # Download as TXT
                agent_report = generate_agent_report(selected_agent, agents_df)
                st.download_button(
                    label="Download as TXT",
                    data=agent_report,
                    file_name=f"{selected_agent}_report.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Download as PDF
                pdf_report = generate_pdf_report(selected_agent, agents_df)
                st.download_button(
                    label="Download as PDF",
                    data=pdf_report,
                    file_name=f"{selected_agent}_report.pdf",
                    mime="application/pdf"
                )
        
        else:
            st.warning("No agents found matching your search criteria.")

    with tab2:
        st.header("📊 Performance Metrics")
        performance_chart = create_agent_performance_chart(agents_df)
        st.plotly_chart(performance_chart, use_container_width=True)
    
    with tab3:
        st.header("💡 Strategic Communication Tactics")
        
        for suggestion in GENERALIZED_SUGGESTIONS:
            st.markdown(f"""
            <div class="card">
                <h3>{suggestion['title']}</h3>
                <p>{suggestion['description']}</p>
                <div style="margin-top: 10px;">
                    <strong>Expected Impact:</strong> {suggestion['impact']}
                </div>
                <div style="margin-top: 10px;">
                    <strong>Key Implementation Techniques:</strong>
                    <ul style="margin-top: 5px;">
                        {''.join(f'<li>{technique}</li>' for technique in suggestion['key_techniques'])}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
        # Download suggestions report
        suggestions_report = generate_suggestions_report(GENERALIZED_SUGGESTIONS)
        st.download_button(
            label="Download Suggestions Report",
            data=suggestions_report,
            file_name="strategic_suggestions_report.txt",
            mime="text/plain"
        )

    with tab4:
        st.header("📊 Market Sentiment Analysis")
        
        # Market Sentiment Pie Chart
        sentiment_chart = market_sentiment_analysis(agents_df)
        st.plotly_chart(sentiment_chart, use_container_width=True)
        
        st.markdown("""
        <div class="card">
            <h3>Understanding Market Sentiment</h3>
            <p>Our advanced sentiment analysis combines client satisfaction and deal closure rates to provide a comprehensive view of agent performance and market dynamics.</p>
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.header("🔮 Property Price Forecasting")
        
        # Property Price Trend Prediction
        price_trend_chart = simulate_property_price_trends()
        st.plotly_chart(price_trend_chart, use_container_width=True)
        
        # Feature Importance Visualization
        st.subheader("Agent Performance Insights")
        feature_importance_fig = feature_importance_visualization(agents_df)
        st.pyplot(feature_importance_fig)
        
        st.markdown("""
        <div class="card">
            <h3>Advanced Predictive Modeling</h3>
            <p>Our platform leverages machine learning techniques to provide:</p>
            <ul>
                <li>Accurate property price predictions</li>
                <li>Feature importance analysis</li>
                <li>Market trend insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Sidebar for Additional Information
def add_sidebar():
    st.sidebar.title("🏘️ Real Estate Intel")
    st.sidebar.markdown("""
    ### Quick Insights
    - Total Agents: 4
    - Total Deals Closed: 200
    - Average Client Satisfaction: 93%
    
    ### Top Performing Agents
    1. Priya Sharma (62 deals)
    2. Sneha Reddy (55 deals)
    3. Rajesh Kumar (45 deals)
    
    ### Market Trends
    - Rising property prices
    - Increased digital interactions
    - Growing investor confidence
    """)

# Main execution
def main_app():
    # Add sidebar
    add_sidebar()
    
    # Run main application
    main()

if __name__ == "__main__":
    main_app()
