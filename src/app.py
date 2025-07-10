import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from transformer_lens import HookedTransformer

# Custom CSS for modern UI
st.markdown('''
    <style>
        .main {background-color: #f0f2f6;}
        .stButton>button {background-color: #4CAF50; color: white;}
        .stSidebar {background-color: #ffffff;}
    </style>
''', unsafe_allow_html=True)

st.title('Interp-Toolkit: LLM Interpretability Dashboard')

# Sidebar navigation
page = st.sidebar.selectbox('Select Page', ['Home', 'Load Activations', 'Visualize', 'Interventions', 'Case Study'])

if page == 'Home':
    st.header('Welcome to Interp-Toolkit')
    st.write('Analyze activations from models like TinyLlama-1.1B and Gemma-2B-It.')
    st.write('Use the sidebar to navigate.')

elif page == 'Load Activations':
    st.header('Load Activation Dumps')
    uploaded_file = st.file_uploader('Upload JSON log', type='json')
    if uploaded_file:
        data = json.load(uploaded_file)
        st.session_state['data'] = data
        st.success('File loaded successfully!')

elif page == 'Visualize':
    st.header('Visualize Activations')
    if 'data' in st.session_state:
        df = pd.DataFrame(st.session_state['data'])
        fig = px.line(df, x='layer', y='activation', title='Activation Patterns')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning('Please load data first.')

elif page == 'Interventions':
    st.header('Perform Interventions')
    if 'data' in st.session_state:
        model_name = st.selectbox('Select Model', ['TinyLlama-1.1B', 'Gemma-2B-It'])
        model = HookedTransformer.from_pretrained(model_name)
        # Simple intervention example
        if st.button('Run Patch'):
            st.write('Patching activations...')
            # Placeholder for intervention logic
            st.success('Intervention complete!')
    else:
        st.warning('Please load data first.')

elif page == 'Case Study':
    st.header('Case Study: Regex-Sink Mitigation')
    st.write('Analyzing regex patterns in activations.')
    # Load sample data if available
    sample_path = '../samples/sample_activations.json'
    if os.path.exists(sample_path):
        with open(sample_path, 'r') as f:
            sample_data = json.load(f)
        df_sample = pd.DataFrame(sample_data)
        fig_sample = px.heatmap(df_sample, title='Sample Activation Heatmap')
        st.plotly_chart(fig_sample, use_container_width=True)
    else:
        st.info('Sample data not found. Upload your own in Load Activations.') 