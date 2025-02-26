import streamlit as st
import webbrowser

# Page Configuration
st.set_page_config(page_title="Ascend Analytics", page_icon="📊", layout="wide")

# Centered Title
st.markdown("""
    <h1 style='text-align: center;'>🌟 Ascend Analytics 🌟</h1>
    <h3 style='text-align: center;'>Optimizing Workforce Management and Business Analytics with cutting-edge tools</h3>
    <hr>
""", unsafe_allow_html=True)

# Services Section
st.subheader("🔧 Our Services")

services = {
    "Forecasting Tool with Workforce Management Calculator": {
        "description": "Helps businesses predict workload, staffing needs, and optimize workforce allocation for improved efficiency.",
        "video": "C:/Users/ryoaki/Downloads/Forecast_Capcalculator.mp4"
    },
    "FTE Capacity Planning Tool (Weekly & Intraday)": {
        "description": "Provides precise weekly and intraday (30-minute interval) workforce planning to ensure optimal staffing levels.",
        "video": "C:/Users/ryoaki/Downloads/Intraday_cap_plan.mp4"
    },
    "KPI Dashboard with AI Assistant": {
        "description": "Displays up to 8 key performance indicators (KPIs) with AI-driven insights to enhance data-driven decision-making.",
        "video": "C:/Users/ryoaki/Downloads/AI KPI Insights.mp4"
    }
}

for service, details in services.items():
    st.markdown(f"""
        <div style='text-align: center;'>
            <h4>{service}</h4>
            <p>{details['description']}</p>
        </div>
    """, unsafe_allow_html=True)

    # Embed local video
    st.video(details["video"])

    st.markdown("<hr>", unsafe_allow_html=True)

# Pricing Section
st.subheader("💲 Pricing")
st.markdown("""
    <div style='text-align: center;'>
        <p><b>Full Access (Includes KPI Dashboard & AI Assistant):</b> $25/user/month</p>
        <p><b>Basic Plan (Excludes KPI Dashboard & AI Assistant):</b> $10/user/month (No Setup Fee)</p>
        <p><b>KPI Dashboard Setup (Up to 8 KPIs):</b> $2,500 (2-3 months implementation)</p>
        <p><b>Free Maintenance:</b> 60 days post-setup</p>
        <p><b>Custom Features:</b> Additional charges based on complexity</p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Booking Section
st.subheader("📅 Book a Demo & Consultation")
st.markdown("<div style='text-align: center;'>Schedule a session to explore our solutions:</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center;'><a href='https://calendly.com/mpailden/30min' target='_blank'>📅 Book an Appointment</a></div>",
    unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div style='text-align: center;'>
        <p>More tools are coming soon! Stay tuned for updates.</p>
        <p>© 2025 Ascend Analytics. All Rights Reserved.</p>
    </div>
""", unsafe_allow_html=True)
