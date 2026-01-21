import streamlit as st
import pandas as pd
from run import get_scf_appointment

scf_appointment=get_scf_appointment()

st.set_page_config(page_title="Temu SCF Appointment", layout="wide")

st.title("SCF拼车规划")
with st.expander("拼车数据", expanded=False):
    st.dataframe(scf_appointment)