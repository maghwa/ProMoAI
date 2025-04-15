import os
import shutil
import subprocess
import streamlit as st
import tempfile

import promoai
from promoai.general_utils.app_utils import InputType, ViewType
from promoai.general_utils.ai_providers import AI_MODEL_DEFAULTS, DEFAULT_AI_PROVIDER, TOGETHER_MODELS, OLLAMA_MODELS, AIProviders
from pm4py import read_xes, read_pnml, read_bpmn, convert_to_petri_net, convert_to_bpmn
from pm4py.util import constants
from pm4py.objects.petri_net.exporter.variants.pnml import export_petri_as_string
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.bpmn import visualizer as bpmn_visualizer
from pm4py.objects.bpmn.layout import layouter as bpmn_layouter
from pm4py.objects.bpmn.exporter.variants.etree import get_xml_string

# Set page configuration
st.set_page_config(
    page_title="Process Modeling with Generative AI",
    page_icon="⭐",
    layout="wide"
)

# Custom CSS with RED theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #8B0000;  /* Dark Red */
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 1rem;
        color: #A52A2A;  /* Brown Red */
    }
    .stButton>button {
        background-color: #B22222;  /* FireBrick Red */
        color: white;
        font-weight: 500;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #8B0000;  /* Dark Red */
    }
    .success-message {
        background-color: #FFEBEE;  /* Light Red */
        color: #B71C1C;  /* Deep Red */
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        border-left: 5px solid #B71C1C;
    }
    .info-box {
        background-color: #FFEBEE;  /* Light Red */
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        border-left: 5px solid #EF5350;  /* Medium Red */
    }
    .feedback-box {
        background-color: #FFCDD2;  /* Slightly deeper light red */
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        border-left: 5px solid #E53935;  /* Medium-Dark Red */
    }
    .st-bx {
        border: 1px solid #FFCDD2;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #FFEBEE;
        border-radius: 4px 4px 0 0;
        color: #B71C1C;
        padding: 10px 20px;
        border: 1px solid #FFCDD2;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EF5350;
        color: white;
    }
    /* Red outline on inputs */
    div[data-baseweb="select"] > div,
    .stSlider [data-baseweb="slider"],
    .stTextInput > div > div,
    .stTextArea > div > div,
    .stFileUploader > div {
        border: 1px solid #FFCDD2;
        border-radius: 4px;
    }
    .stTextInput > div:focus-within, 
    .stTextArea > div:focus-within {
        border-color: #E53935;
    }
    /* Expander header */
    .streamlit-expanderHeader {
        background-color: #FFEBEE;
        border: 1px solid #FFCDD2;
        color: #B71C1C;
    }
    /* Red theme for progress bar */
    .stProgress > div > div {
        background-color: #EF5350;
    }
</style>
""", unsafe_allow_html=True)

def run_app():
    from promoai.general_utils.ai_providers import TOGETHER_MODELS, OLLAMA_MODELS, AIProviders

    st.markdown('<div class="main-header">🔥 Process Modeling with Generative AI</div>', unsafe_allow_html=True)
    
    # About section in expander
    with st.expander("ℹ️ About this application"):
        st.markdown("""
        This application uses generative AI to automatically create process models from textual descriptions. 
        It leverages both local and cloud-based AI models, allowing you to:
        
        - Generate BPMN process models from text descriptions
        - Provide feedback to refine the generated models
        - Export models in standard formats (BPMN, PNML)
        
        This tool was developed as part of an internship project focused on improving business process modeling through AI.
        
        **Available Models:**
        
        Ollama (Local):
        - **Gemma 3 (4B)**: Good for simple processes
        - **DeepSeek Coder**: Best for complex processes with many activities
        - **DeepCoder**: Optimized for logical structures and decision flows
        
        Together AI (Cloud):
        - **Llama 3 (70B)**: High-quality process modeling for complex scenarios
        - **Llama 4 Scout (17B)**: Latest model with enhanced reasoning for processes
        - **Mixtral 8x7B**: Balanced performance for various process types
        """)
    
    # Initialize session state
    if 'model_gen' not in st.session_state:
        st.session_state['model_gen'] = None
    if 'feedback' not in st.session_state:
        st.session_state['feedback'] = []
    if 'selected_mode' not in st.session_state:
        st.session_state['selected_mode'] = "Model Generation"
    if 'ai_provider' not in st.session_state:
        st.session_state['ai_provider'] = AIProviders.TOGETHER.value

    # Configuration section
    st.markdown('<div class="sub-header">⚙️ Configuration</div>', unsafe_allow_html=True)

    from promoai.general_utils.ai_providers import TOGETHER_MODELS, OLLAMA_MODELS, AIProviders

    # Provider selection
    ai_provider = st.selectbox(
        "Select AI Provider:",
        options=[AIProviders.OLLAMA.value, AIProviders.TOGETHER.value],
        index=0,
        help="Choose between local Ollama models or cloud-based Together AI models"
    )

    # Show model selection dropdown based on provider
    if ai_provider == AIProviders.TOGETHER.value:
        model_name = st.selectbox(
            "Select Together AI Model:",
            options=list(TOGETHER_MODELS.keys()),
            index=0,
            format_func=lambda x: f"{x.split('/')[-1]} - {TOGETHER_MODELS[x]}",
            help="Select the model to use for process modeling"
        )
        
        # API Key input
        api_key = st.text_input(
            "Together API Key:",
            type="password",
            placeholder="Enter your Together API key here",
            help="Get your API key from together.ai"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your Together API key to use the service.")
    else:
        model_name = st.selectbox(
            "Select Ollama Model:",
            options=list(OLLAMA_MODELS.keys()),
            index=0,
            format_func=lambda x: f"{x} - {OLLAMA_MODELS[x]}",
            help="Select the model to use for process modeling"
        )
        api_key = ""  # Not needed for Ollama
    
    # Input type selection - MOVED OUTSIDE THE ELSE BLOCK
    st.markdown('<div class="sub-header">📝 Input Selection</div>', unsafe_allow_html=True)
    
    input_type = st.radio(
        "Select Input Type:",
        options=[InputType.TEXT.value, InputType.MODEL.value, InputType.DATA.value], 
        horizontal=True,
        index=0
    )
    
    if input_type != st.session_state['selected_mode']:
        st.session_state['selected_mode'] = input_type
        st.session_state['model_gen'] = None
        st.session_state['feedback'] = []
        st.rerun()
    
    # Input form
    st.markdown('<div class="sub-header">🔍 Process Specification</div>', unsafe_allow_html=True)
    
    with st.form(key='model_gen_form'):
        if input_type == InputType.TEXT.value:
            st.markdown('<div class="info-box">Describe your business process in natural language. Be specific about activities, decision points, and flow relationships.</div>', unsafe_allow_html=True)
            
            description = st.text_area(
                "Process Description:", 
                height=200,
                placeholder="Example: A customer orders a product online. After the order is received, payment processing and inventory check happen in parallel. If the product is available and payment is successful, the order is shipped. Otherwise, the customer is notified about the issue..."
            )
            
            submit_button = st.form_submit_button(label='Generate Process Model')
            if submit_button:
                if ai_provider == AIProviders.TOGETHER.value and not api_key:
                    st.error("⚠️ Please enter your Together API key to use the Together AI service.")
                else:
                    with st.spinner("🧠 The AI is creating your process model..."):
                        try:
                            process_model = promoai.generate_model_from_text(
                                description,
                                api_key=api_key,
                                ai_model=model_name,
                                ai_provider=ai_provider
                            )
                            
                            st.session_state['model_gen'] = process_model
                            st.session_state['feedback'] = []
                        except Exception as e:
                            st.error(body=str(e), icon="⚠️")
                            pass
        
        elif input_type == InputType.DATA.value:
            st.markdown('<div class="info-box">Upload an event log in XES format to discover a process model.</div>', unsafe_allow_html=True)
            
            uploaded_log = st.file_uploader(
                "Upload event log (XES):",
                type=["xes", "xes.gz"]
            )
            
            submit_button = st.form_submit_button(label='Discover Process Model')
            if submit_button:
                if uploaded_log is None:
                    st.error(body="Please upload an event log file first", icon="⚠️")
                    pass
                
                with st.spinner("⏳ Discovering process model from event log..."):
                    try:
                        temp_dir = "temp"
                        contents = uploaded_log.read()
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        with tempfile.NamedTemporaryFile(mode="wb", delete=False,
                                                         dir=temp_dir, suffix=uploaded_log.name) as temp_file:
                            temp_file.write(contents)
                            log = read_xes(temp_file.name, variant="rustxes")
                        
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        process_model = promoai.generate_model_from_event_log(log)
                        st.session_state['model_gen'] = process_model
                        st.session_state['feedback'] = []
                    except Exception as e:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        st.error(body=f"Error during discovery: {e}", icon="⚠️")
                        st.stop()
        
        elif input_type == InputType.MODEL.value:
            st.markdown('<div class="info-box">Upload an existing BPMN or Petri net model to refine it.</div>', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload existing process model:",
                type=["bpmn", "pnml"]
            )
            
            submit_button = st.form_submit_button(label='Load Model')
            if submit_button:
                if uploaded_file is None:
                    st.error(body="Please upload a model file first", icon="⚠️")
                    pass
                
                with st.spinner("⏳ Loading and processing the model..."):
                    try:
                        temp_dir = "temp"
                        file_extension = uploaded_file.name.split(".")[-1].lower()
                        contents = uploaded_file.read()
                        
                        os.makedirs(temp_dir, exist_ok=True)
                        if file_extension == "bpmn":
                            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bpmn",
                                                            dir=temp_dir) as temp_file:
                                temp_file.write(contents)
                                bpmn_graph = read_bpmn(temp_file.name)
                                process_model = promoai.generate_model_from_bpmn(bpmn_graph)
                                
                        elif file_extension == "pnml":
                            with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pnml",
                                                            dir=temp_dir) as temp_file:
                                temp_file.write(contents)
                                pn, im, fm = read_pnml(temp_file.name)
                                process_model = promoai.generate_model_from_petri_net(pn)
                                
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        st.session_state['model_gen'] = process_model
                        st.session_state['feedback'] = []
                    except Exception as e:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                        st.error(body=f"Error processing model: {e}", icon="⚠️")
    
    # Display the generated model
    if 'model_gen' in st.session_state and st.session_state['model_gen']:
        st.markdown('<div class="success-message">✅ Process model successfully generated!</div>', unsafe_allow_html=True)
        
        # Results section
        st.markdown('<div class="sub-header">🔄 Model Results</div>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Visualization & Export", "Feedback & Refinement"])
        
        try:
            with tab1:
                # Visualization options
                view_option = st.selectbox(
                    "Select visualization format:", 
                    [v_type.value for v_type in ViewType],
                    index=0
                )
                
                # Prepare visualization
                process_model_obj = st.session_state['model_gen']
                powl = process_model_obj.get_powl()
                pn, im, fm = convert_to_petri_net(powl)
                bpmn = convert_to_bpmn(pn, im, fm)
                bpmn = bpmn_layouter.apply(bpmn)
                
                # Generate visualization
                image_format = "svg"
                if view_option == ViewType.POWL.value:
                    from pm4py.visualization.powl import visualizer
                    vis_str = visualizer.apply(powl, parameters={'format': image_format})
                elif view_option == ViewType.PETRI.value:
                    visualization = pn_visualizer.apply(pn, im, fm, parameters={'format': image_format})
                    vis_str = visualization.pipe(format='svg').decode('utf-8')
                else:  # BPMN
                    from pm4py.objects.bpmn.layout import layouter
                    layouted_bpmn = layouter.apply(bpmn)
                    visualization = bpmn_visualizer.apply(layouted_bpmn, parameters={'format': image_format})
                    vis_str = visualization.pipe(format='svg').decode('utf-8')
                
                # Display visualization in a container with border
                st.markdown("<div style='border:1px solid #FFCDD2; padding:10px; border-radius:4px;'>", unsafe_allow_html=True)
                st.image(vis_str)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Export options
                st.markdown("### Export Options")
                download_1, download_2 = st.columns(2)
                
                with download_1:
                    bpmn_data = get_xml_string(bpmn, parameters={"encoding": constants.DEFAULT_ENCODING})
                    st.download_button(
                        label="📥 Download BPMN",
                        data=bpmn_data,
                        file_name="process_model.bpmn",
                        mime="application/xml"
                    )
                
                with download_2:
                    pn_data = export_petri_as_string(pn, im, fm)
                    st.download_button(
                        label="📥 Download PNML",
                        data=pn_data,
                        file_name="process_model.pnml",
                        mime="application/xml"
                    )
                
                # Show model code
                with st.expander("View Generated Python Code"):
                    st.code(process_model_obj.get_code(), language="python")
            
            with tab2:
                st.markdown('<div class="feedback-box">💬 Provide feedback to refine the model</div>', unsafe_allow_html=True)
                
                # Feedback form
                with st.form(key='feedback_form'):
                    feedback = st.text_area(
                        "Suggest improvements or corrections:",
                        placeholder="Example: Add a decision point after payment verification...",
                        height=150
                    )
                    
                    if st.form_submit_button(label='🔄 Update Model'):
                        if ai_provider == AIProviders.TOGETHER.value and not api_key:
                            st.error("⚠️ Please enter your Together API key to use the Together AI service.")
                        else:
                            with st.spinner("🔄 Refining the process model..."):
                                try:
                                    process_model = st.session_state['model_gen']
                                    process_model.update(
                                        feedback, 
                                        api_key=api_key,
                                        ai_model=model_name,
                                        ai_provider=ai_provider
                                    )
                                    st.session_state['model_gen'] = process_model
                                    st.session_state['feedback'].append(feedback)
                                    st.success("Model updated successfully!")
                                except Exception as e:
                                    st.error(f"Update failed: {e}")
                
                # Display feedback history
                if len(st.session_state['feedback']) > 0:
                    st.markdown("### Feedback History")
                    for i, fb in enumerate(st.session_state['feedback']):
                        st.markdown(f"**Feedback #{i+1}:**")
                        st.markdown(f"<div style='background-color:#FFCDD2; padding:10px; border-radius:4px; border-left:5px solid #E53935;'>{fb}</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error displaying model: {e}")

    # Footer
    st.markdown("""
    <div style="position:fixed; bottom:0; width:100%; background-color:#FFEBEE; padding:10px; text-align:center; border-top:1px solid #FFCDD2;">
        <p style="color:#B71C1C; margin:0;">
            Developed as part of an internship project on applying AI to business process modeling
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add space to account for footer
    st.markdown("<div style='margin-bottom:40px'></div>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    run_app()