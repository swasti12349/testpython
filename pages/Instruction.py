

import streamlit as st
st.set_page_config(page_title="Instructions", page_icon="nathcorp.jpg")

st.title("📖 Instructions for Using the Resume Filter App", anchor=False)
# st.set_page_config(page_title="adasd", page_icon="nathcorp.jpg")
st.sidebar.page_link("pages/home.py", label="Home")
st.sidebar.page_link("pages/instruction.py", label="Instruction")


st.markdown("""
**Overview**
The **Resume Filter App** helps recruiters and hiring managers efficiently screen resumes using AI-powered analysis. It allows you to define evaluation criteria, input a job description, and upload resumes to generate a ranked list of candidates based on skill-match scoring.

---

 

**1. Set Criteria Weightage**
- Navigate to the **sidebar** and assign weightage (%) to each filtering category:
  - Skills
  - Experience
  - Education
  - Soft Skills
- The **total weightage must equal 100%**.
- If you wish to skip a criterion, simply assign it a value of **0**.

**2. Enter Job Description**
- Type or paste the **Job Description (JD)** into the provided text area.
- JD **must not be blank** and should contain **at least 10 words**.
- Include key responsibilities, required skills, and qualifications for best results.

**3. Upload Resumes**
- Click the **"Browse Files"** button to upload one or multiple resumes.
- Supported formats: **PDF**, **DOCX**, **DOC** and **TXT**.
- Ensure resumes are well-structured for accurate parsing.

**4. Reset Selected Resumes (Optional)**
- If you wish to **clear all uploaded resumes**, click the **"Reset Selected Resumes"** button.
- This will remove all previously selected files, allowing you to upload new ones.

**5. Submit for Analysis**
- Click the **"Submit"** button to start the resume filtering process.
- The app will analyze each resume against the JD and weightage criteria.
- Please wait until the analysis completes.

**6. View Results**
- A **summary of insights** and a **detailed result table** will be generated.
- The table includes skill-match scores and other key indicators.
- You can **download the results as a CSV** for further review.

**7. Logout**
- To exit the session, click the **"Logout"** button in the sidebar.

---

**Tips for Best Results**
- Use detailed and well-structured JDs with clearly listed skills.
- Make sure resumes are not scanned images and follow standard formatting.
- Balance weightage distribution to reflect realistic hiring priorities.

""")
