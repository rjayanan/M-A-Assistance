import streamlit as st
from backend import (
    get_ticker_symbol,
    get_stock_data,
    display_stock_chart,
    ai_request,
    extract_tickers
)
import re

# Streamlit App
st.title("M&A Assistant")
st.sidebar.title("Navigation")
action_type = st.sidebar.radio("What do you need help with?", ["Help with Acquiring", "Help with Merging"])

st.subheader("AI-Powered Mergers and Acquisitions Assistance")

company_name = st.text_input("Enter your company's name:", help="Enter the company name (e.g., Apple, Microsoft, Tesla)")

if company_name:
    if action_type == "Help with Acquiring":
        st.header(f"Acquisition Assistance for {company_name}")

        ticker_symbol, formal_name = get_ticker_symbol(company_name)
    
        if ticker_symbol:
            st.success(f"Found ticker symbol: {ticker_symbol} for {formal_name}")
            
            main_stock_data = get_stock_data(ticker_symbol)
            if main_stock_data is not None:
                display_stock_chart(main_stock_data, formal_name, ticker_symbol)
        else:
            st.error(f"Could not find ticker symbol for {company_name}.")
        
        # Fetch and display AI data for the main company
        company_data_message = f"Provide an overview of the company {company_name}, including its current market position and predicted growth or future performance."
        company_data = ai_request(company_data_message)
        st.subheader(f"Overview of {company_name}")
        st.write(company_data)

        # Get AI recommendations for acquisition
        acquisition_message = f"Suggest a list of companies that {company_name} could acquire. Include both startups and larger companies. Go into deep financial analysis of why {company_name} should acquire these companies.Companies must include their name and ticker."
        acquisition_suggestions = ai_request(acquisition_message)
        st.subheader(f"AI Recommendations for Companies to Acquire for {company_name}")
        st.write(acquisition_suggestions)

        pattern = r"^(\d+)\.\s+([A-Za-z\s&,.]+)\s\(([A-Z]+)\)"
        matches = re.findall(pattern, acquisition_suggestions, re.MULTILINE)
        for idx, company, tick in matches:
            st.write(f"**{company}**")
            main_stock_data = get_stock_data(tick)
            display_stock_chart(main_stock_data, company, tick)

    elif action_type == "Help with Merging":
        st.header(f"Merger Assistance for {company_name}")
        ticker_symbol, formal_name = get_ticker_symbol(company_name)
    
        if ticker_symbol:
            st.success(f"Found ticker symbol: {ticker_symbol} for {formal_name}")
            
            main_stock_data = get_stock_data(ticker_symbol)
            if main_stock_data is not None:
                display_stock_chart(main_stock_data, formal_name, ticker_symbol)
        else:
            st.error(f"Could not find ticker symbol for {company_name}.")
            
        # Fetch and display AI data for the main company
        company_data_message = f"Provide an overview of the company {company_name}, including its current market position and predicted growth or future performance."
        company_data = ai_request(company_data_message)
        st.subheader(f"Overview of {company_name}")
        st.write(company_data)

        # Get AI recommendations for merging
        merger_message = f"Suggest a list of companies that might be interested in merging with {company_name}. Include both startups and larger companies. Go into deep financial analysis of why {company_name} should merge these companies.Companies must include their name and ticker."
        merger_suggestions = ai_request(merger_message)
        st.subheader(f"AI Recommendations for Potential Mergers for {company_name}")
        st.write(merger_suggestions)

        pattern = r"^(\d+)\.\s+([A-Za-z\s&,.]+)\s\(([A-Z]+)\)"
        matches = re.findall(pattern, merger_suggestions, re.MULTILINE)
        for idx, company, tick in matches:
            st.write(f"**{company}**")
            main_stock_data = get_stock_data(tick)
            display_stock_chart(main_stock_data, company, tick)

# Footer
st.sidebar.info("Built using Streamlit, Groq AI, and Yahoo Finance.")