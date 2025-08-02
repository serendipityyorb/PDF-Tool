import streamlit as st
from langchain.memory import ConversationBufferMemory
from PDFQA import GetAIAnswer
st.title("% PDF Asking and Answering ")
with st.sidebar:
    api_key = st.text_input("Please input api key:",type="password")
    base_url = st.text_input("Please api port:")
    st.markdown("[click here to get tutor](https://ai.nengyongai.cn/study)")
    st.markdown("<br>"*2,unsafe_allow_html=True)
    st.markdown("&nbsp;"*15+"**Author**"+"&nbsp;"*3+"*Tulip*", unsafe_allow_html=True)
    st.markdown("&nbsp;"*15+"*if a sequence is monotone and bounded, then it converges.*",unsafe_allow_html=True)


st.markdown("##### Please input your pdf file here.")
pdf_file = st.file_uploader("",type="pdf")
question = st.text_input("Please input your question here.",disabled=not pdf_file)

submit_button = st.button("submit",disabled= not question)

if submit_button and not api_key:
    st.info("Remember API Key!")
    st.stop()
if submit_button and not base_url:
    st.info("Remember API Port!")
    st.stop()
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True,memory_key="chat_history",output_key="answer")
if submit_button:
    with st.spinner("OK,I'll thinking carefully."):
        response = GetAIAnswer("gpt-4o-mini",api_key,base_url,pdf_file,st.session_state["memory"],question)
        st.markdown("### Answer as following:")
        st.write(response["answer"])
        st.divider()

    with st.expander("History Message"):
        for i in range(0,len(response["chat_history"]),2):
            human_message = response["chat_history"][i]
            ai_message = response["chat_history"][i+1]
            st.chat_message("human").write(human_message.content)
            st.chat_message("ai").write(ai_message.content)
            st.divider()







